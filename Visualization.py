import visdom
import numpy as np
import torch
from time import sleep


class VisdomPlotter:
    def __init__(self):
        self.viz = visdom.Visdom()
        self.viz_plot1_x = dict()

    def plot_images(self, title, images, caption, keep_history):
        #max = images.max()
        #min = images.min()
        #if (max - min > 0.0000001):
        #    images = (images - min) / (max - min)
        images = (images+1.0)/2.0
        images = torch.clamp(images, 0.0, 1.0)
        self.viz.images(tensor=images,
                        nrow=4,
                        padding=1,
                        win=title,
                        opts=dict(title=caption, width=1024, height=356, jpgquality=70, store_history=keep_history))  #
        #sleep(0.01)  # visdom uses a bad random function in _send to generate window ids which might return the same id if windows creation request are sent too fast

    def plot_line(self, title, value, caption, do_clip_vals=True, clip_val=2.0):
        if do_clip_vals is True:
            if value > clip_val:
                value = clip_val
            if value < -clip_val:
                value = -clip_val
        if title in self.viz_plot1_x:
            x = self.viz_plot1_x[title] + 1
            self.viz_plot1_x[title] = x
        else:
            self.viz_plot1_x[title] = 1
            x = 1
        self.viz.line(
                X=torch.ones((1, 1)).cpu() * x,
                Y=torch.Tensor([value]).unsqueeze(0).cpu(),
                win=title,
                update='append',
                opts=dict(
                    fillarea=True,
                    linecolor=np.array([[255, 0, 0],]),
                    dash=np.array(['solid']),
                    showlegend=False,
                    title=caption,
                    marginleft=30,
                    marginright=30,
                    marginbottom=80,
                    margintop=30,
                )
            )
        #sleep(0.01)

    def plot_lines(self, title, caption, step, values, legends, colors):
        arr = np.array(values)
        ten = torch.from_numpy(arr)
        arr_colors = np.array(colors)
        X = torch.ones((1,ten.shape[0])).cpu() * step
        Y = ten.unsqueeze(0).cpu()
        self.viz.line(
            X=X,
            Y=Y,
            win=title,
            update='append',
            opts=dict(
                ytype='log',
                fillarea=True,
                linecolor=arr_colors,  # np.array([[255, 0, 0], ]),
                #dash=np.array(['solid']),
                showlegend=True,
                legend= legends,
                title=caption,
                marginleft=30,
                marginright=30,
                marginbottom=80,
                margintop=30,
                markers=False,
                markersymbol='dot',
                markersize=10,
            )
        )
