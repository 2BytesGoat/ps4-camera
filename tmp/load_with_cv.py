import cv2
import glob
import os
import torch
import utils
from torch.autograd import Variable
from torchvision.transforms import Compose

from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet


if __name__ == '__main__':
    optimize    = True
    input_path  = "input"
    output_path = "output" 
    model_path  = "weights"
    model_name       = "midas_v21_small-70d6b9c8.pt"
    torch_model_path = os.path.join(model_path, model_name)
    cv2_model_name   = "model-small.onnx"
    cv2_model_path   =  os.path.join(model_path, model_name)

    torch.cuda.empty_cache()
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    original_model = MidasNet_small(torch_model_path, 
                                    features=64, 
                                    backbone="efficientnet_lite3", 
                                    exportable=True, 
                                    non_negative=True, 
                                    blocks={'expand': True})
    net_w, net_h = 256, 256
    resize_mode="upper_bound"
    normalization = NormalizeImage(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = Compose(
    [
        Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method=resize_mode,
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        normalization,
        PrepareForNet(),
    ]
    )

    original_model.eval()
    original_model.to(device)

    if optimize==True:
        # rand_example = torch.rand(1, 3, net_h, net_w)
        # model(rand_example)
        # traced_script_module = torch.jit.trace(model, rand_example)
        # model = traced_script_module
    
        if device == torch.device("cuda"):
            original_model = original_model.to(memory_format=torch.channels_last)  
            original_model = original_model.half()

    # # model export into ONNX format
    # torch.onnx.export(
    #     original_model,
    #     generated_input,
    #     full_model_path,
    #     verbose=False,
    #     input_names=["input"],
    #     output_names=["output_conv"],
    #     opset_version=11
    # )

    # read converted .onnx model with OpenCV API
    opencv_net = cv2.dnn.readNetFromONNX(cv2_model_path)

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input

        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if optimize==True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)  
                sample = sample.half()
            # prediction = original_model.forward(sample)
            # prediction = (
            #     torch.nn.functional.interpolate(
            #         prediction.unsqueeze(1),
            #         size=img.shape[:2],
            #         mode="bicubic",
            #         align_corners=False,
            #     )
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            opencv_net.setInput(sample.squeeze()
                                .cpu()
                                .numpy())
            prediction = opencv_net.forward()


        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        utils.write_depth(filename, prediction, bits=2)

    print("finished")