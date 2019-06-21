import torch

from blazeface import BlazeFace, MediaPipeBlazeFace


def test_forward():
    x = torch.randn(1, 3, 128, 128)
    model = BlazeFace()
    h = model(x)

    assert h.detach().numpy().shape == (1, 96, 8, 8)


def test_forward_original():
    x = torch.randn(1, 3, 128, 128)
    model = MediaPipeBlazeFace()
    h = model(x)

    assert h.detach().numpy().shape == (1, 96, 8, 8)


def test_export_onnx(tmpdir):
    p = str(tmpdir.join('blazeface.onnx'))

    x = torch.randn(1, 3, 128, 128)
    model = BlazeFace()
    torch.onnx.export(model, x, p, verbose=True,
                      input_names=['input'],
                      output_names=['output'])
