import os
import json
import time

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnetv2_s as create_model


def main():
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = "/hpc2hdd/home/cguo847/5013/flower_photos/daisy/5547758_eea9edfd54_n.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=5).to(device)
    # load model weights
    model_weight_path = "/hpc2hdd/home/cguo847/weights/model-flower-small-batch512.pth"
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # model.eval()
    # 加载预训练权重
    state_dict = torch.load(model_weight_path, map_location=device)
    # 过滤掉分类器层的权重
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.classifier')}
    
    # 加载其他权重
    model.load_state_dict(state_dict, strict=False)
    
    # 重新初始化分类器层
    in_features = model.head.classifier.in_features
    model.head.classifier = torch.nn.Linear(in_features, 5).to(device)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

    # 记录结束时间并计算总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.3f} seconds ({total_time * 1000:.0f} milliseconds)")


if __name__ == '__main__':
    main()