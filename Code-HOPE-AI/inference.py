import os
import torch
import argparse
from os.path import join as path_join

from lib.admil import CenterLoss
from lib.pvtv2_lstm import LSTMModel

from utils.utils import inference_preprocess, inference_one_bag

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="HOPE AI inference parser")
  parser.add_argument("--case", 
                            type=str,
                            default="Negative_1"
                          )

  args = parser.parse_args()


  this_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = path_join(this_dir, "data")
  weight_path = path_join(this_dir, "weights")
  model_weights_path = path_join(weight_path, "model.pth")
  center_loss_weight_path = path_join(weight_path, "center_loss.pth")
  center_loss_img_weight_path = path_join(weight_path, "center_loss_img.pth")  

  # Model Loading...
  #Model initiazation
  model = LSTMModel() #LSTM
  model_center_loss = CenterLoss(2, 256).cuda()
  model_center_loss_img = CenterLoss(2, 512).cuda()

  model.load_state_dict(torch.load(model_weights_path, map_location="cpu"),strict=False)
  model_center_loss.load_state_dict(torch.load(center_loss_weight_path,map_location="cpu"),strict=False)
  model_center_loss_img.load_state_dict(torch.load(center_loss_img_weight_path,map_location="cpu"),strict=False)

  model.cuda().eval()
  model_center_loss.cuda().eval()
  model_center_loss_img.cuda().eval()

  # data load
  cases_path = path_join(data_dir, "cases")
  case2_use = args.case
  data_path = path_join(cases_path, case2_use) 
  images, names = inference_preprocess(data_path, testsize=352)
  bag_tensor = torch.stack(images).cuda()  # shape: [N, 3, 256, 256]

  # save result path
  save_img_name = data_path.replace('cases', 'results') + '_res.png'
  os.makedirs(os.path.dirname(save_img_name), exist_ok=True)

  with torch.no_grad():
    preds_patient, probs_patient, bag_len = inference_one_bag(model, model_center_loss, model_center_loss_img, bag_tensor, names, topk = 7, save_img_name=save_img_name)
    print(preds_patient)
    print(torch.stack(preds_patient).float().mean())
    pred_probs = sum(x[0] for x in probs_patient) / len(probs_patient)
    pred_vote = torch.stack(preds_patient).float().mean() > 0.5
    print(pred_vote)
    print(probs_patient)
    print(pred_probs)

  print('sample path:', data_path)
  print('predict result:', 'HP - positive' if pred_vote else 'HP - negative')
  print('Visualization saved to:', save_img_name)