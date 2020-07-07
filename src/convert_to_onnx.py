import io
import os
import numpy as np
import torch
import onnx
import onnxruntime
from onnx import optimizer
import net.erfnet as net
from options.options import parser
from options.config import cfg

def load_model():
    global args
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    args.gpus = len(args.gpus)

    num_ego = cfg.NUM_EGO
    num_class = cfg.NUM_CLASSES
    model = net.ERFNet(num_class, num_ego).cuda()
    # model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loaded checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(k, model_state_dict[k].shape,
                                                  state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    return model

def create_onnx_model(args):
    net = load_model()
    net.eval()

    onnx_file = args.onnx_file
    print("==> Exporting model to ONNX format at '{}'".format(onnx_file))
    input_names = ["input"]
    if cfg.NUM_CLASSES and cfg.NUM_EGO:
        output_names = ["output/cls", "output/ego", "output/exist"]
    elif cfg.NUM_CLASSES and cfg.NUM_EGO == 0:
        output_names = ["output/cls"]
    elif cfg.NUM_CLASSES == 0 and cfg.NUM_EGO:
        output_names = ["output/ego", "output/exist"]
    torch_in = torch.randn(1, 3, cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH, device='cuda')

    from thop import profile
    flops, params = profile(net, inputs=(torch_in,))
    from thop import clever_format
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params, flops)

    with torch.no_grad():
        with io.BytesIO() as f:
            torch_out = net(torch_in)
            torch.onnx.export(net,
                              torch_in,
                              f,
                              export_params=True,
                              verbose=False,
                              input_names=input_names,
                              output_names=output_names)
            onnx_model = onnx.load_from_string(f.getvalue())

    all_passes = optimizer.get_available_passes()
    passes = args.onnx_optim_passes.split(',')
    assert all(p in all_passes for p in passes)
    onnx_model = optimizer.optimize(onnx_model, passes)

    onnx.save(onnx_model, onnx_file)

    # import netron
    # netron.start(onnx_file)

    return torch_in, torch_out


def validate_model(args, model_in, model_out):
    ort_session = onnxruntime.InferenceSession(args.onnx_file)

    def to_numpy(tensor):
        if isinstance(tensor, list):
            tensor = tensor[0]
        return tensor.detach().cpu().numpy(
        ) if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(model_in)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results

    for i in range(len(ort_outs)):
        print(ort_outs[i].shape)
        np.testing.assert_allclose(to_numpy(model_out),
                                   ort_outs[i],
                                   rtol=1e-03,
                                   atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, "
          "and the result looks good!")


if __name__ == '__main__':
    args = parser.parse_args()
    inputs, outputs = create_onnx_model(args)
    print("Validating model... ")
    validate_model(args, inputs, outputs)
    print("All Done")
