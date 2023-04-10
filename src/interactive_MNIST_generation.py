import numpy as np
import torch
from config import  MODEL_PATH
import cv2

def drawfunction(event,x,y,flags,param):
    global img, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
    
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.circle(img,(int(x/10),int(y/10)),1,1,-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        img = np.zeros_like(img)

if __name__ == "__main__":

    model = torch.load(MODEL_PATH).to("cpu")

    cv2.namedWindow("draw")
    cv2.setMouseCallback("draw",drawfunction)
    img = np.zeros((28,28,1), dtype=np.float)
    drawing = False
    while(1):
        tmp = img.copy()
        if np.max(tmp) > 0:
            tmp = tmp / np.max(tmp)
        tmp = torch.Tensor(tmp)
        tmp = torch.squeeze(tmp)

        out = model.forward(torch.unsqueeze(tmp, 0))
        out = out.view(28,28,1)
        out = out.detach()
        out = out.numpy()
        out = cv2.resize(out, None, fx=10, fy=10)
        img_view = cv2.resize(img, None, fx=10, fy=10)
        cv2.imshow("out", out)
        cv2.imshow("draw", img_view)
        key = cv2.waitKey(1)
        if key == 27:
            break

        


    