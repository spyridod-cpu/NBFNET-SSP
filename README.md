# NBFNET-SSP: Neural Bellman-Ford Network for solving the Single Source Shortest Path Problem
This is a codebase based on the official codebase of the paper 
[Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction][paper]

[Zhaocheng Zhu](https://kiddozhu.github.io),
[Zuobai Zhang](https://oxer11.github.io),
[Louis-Pascal Xhonneux](https://github.com/lpxhonneux),
[Jian Tang](https://jian-tang.com)

[paper]: https://arxiv.org/pdf/2106.06935.pdf
applying the Neural Bellman-Ford Network to solve the single source shortest path problem.

## Overview ##
The NBFNET-SSP is a graph neural network model based on the NBFNet that aims
to extend the range of the message passing operation and heuristically find the optimal
path from one node to another. This version only tackles the shortest path problem. 
## Installation ##
To install the dependencies run the instructions below. It is recommended that you use 
the exact packages given here and that you use Python 3.8, so that you do not run into problems with compatibility.
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch-lts
pip install torch-cluster==1.5.9 torch-scatter==2.0.8 torchdrug
pip install ogb easydict pyyaml osmnx
```

## Reproduction ##
To reproduce the results of NBFNET-SSP, use the following command to train the model.
```
python scripts/run.py -c config/homogeneous_graph/lagadas_extra_small.yaml --gpus null --version v1
```
We provide the hyperparameters for each experiment in configuration files. All the configuration files can be found in config/*/*.yaml.

## Visualize Paths ##
Once you have models trained on your chosen dataset, you can visualize paths with 
the following line. Please replace the checkpoint parameter with your own path.
```
python scripts/visualize.py -c config/homogeneous_graph/lagadas_extra_small_visualize.yaml --gpus null --checkpoint /path/to/nbfnet-ssp/experiment/model_epoch_20.pth
```

## Results ##
The results of NBFNET-SSP on the datasets used are shown here. All the results are 
obtained using a AMD 5600 cpu. 

<table>
    <tr>
        <th>Dataset</th>
        <th>Number Of Layers</th>
        <th>Number of Iterations</th>
        <th>Accuracy</th>
        <th>Mean Distance from Shortest Path (MDSP)</th>
        <th>MDSP with error recovery</th>
    </tr>
    <tr>
        <th>lagadas_extra_small</th>
        <th>4</th>
        <th>3</th>
        <th>90%</th>
        <th>3 meters</th>
        <th>218 meters</th>
    </tr>
    <tr>
        <th>lagadas_small</th>
        <th>2</th>
        <th>4</th>
        <th>91%</th>
        <th>3 meters</th>
        <th>1370 meters</th>
    </tr>
    <tr>
        <th>evag</th>
        <th>2</th>
        <th>4</th>
        <th>91%</th>
        <th>23 meters</th>
        <th>1210 meters</th>
    </tr>
        <tr>
        <th>center</th>
        <th>4</th>
        <th>30</th>
        <th>75%</th>
        <th>2643 meters</th>
        <th>11982 meters</th>
    </tr>
    <tr>
        <th>auth_small</th>
        <th>4</th>
        <th>20</th>
        <th>83%</th>
        <th>1652 meters</th>
        <th>15629 meters</th>
    </tr>
    
</table>

## Custom Datasets ##
To create your own custom dataset you these steps. 

First you need to decide on a location for your road network. For this purpose head
over to [Open Street Map](https://www.openstreetmap.org) find the location you would like and get the coordinates of 
the bounding box. 

Then you need to create a yaml configuration file create_name_of_your_dataset.yaml, like
the ones in the config folder. You can replace the name with the chosen name of your 
dataset and the coordinates of the bounding box with your own. Note that the name
you choose must be consistent across all configuration files. 
Having done that you can run the following command.

```commandline
python scripts/create_dataset.py -c config/homogeneous_graph/create_name_of_your_dataset.yaml
```
This script creates the files storing the nodes, edges and weights your chosen location.
Next, you need to create your own configuration files for training and visualization.

For training, copy the configuration file lagadas_extra_small.yaml into your own name_of_dataset.yaml,
and change the fields path, name and num_nodes to correspond to the location of the data folder, the name of your dataset
and the number of nodes of your graph. You can find the number of nodes by heading to the data folder and opening the file
name_of_your_dataset.txt and getting the line number of the file.
The hyperparameters you can play with are input_dims, hidden_dims, num_iterations and num_epoch.
The rest should be left untouched.

Then you can run the familiar command:
```
python scripts/run.py -c config/homogeneous_graph/name_of_your_dataset.yaml --gpus null --version v1
```

To visualize the results you need to create a final configuration file.
You can copy the contents of the yaml file you just created and remove the field "train".
Then, you need to add the following lines at the end.

```yaml
 checkpoint: {{ checkpoint }}

 data:
  name: name_of_your_dataset
  bounding_box:
   xlow: x_south
   xhigh: x_north
   ylow: y_south
   yhigh: y_north
  threshold: 0
```

The other fields must match the fields of your training configuration, except the field "recover".
This field declares if an error recovery algorithm is employed. The threshold hyperparameter determines
the minimum distance of a path produced by the model to the shortest path for it to be saved. A threshold
of 0 means that all paths are plotted. These figures are saved in the "figures" folder. 

Then you can run the familiar command.
```
python scripts/visualize.py -c config/homogeneous_graph/name_of_your_dataset.yaml --gpus null --checkpoint /path/to/nbfnet-ssp/experiment/model_epoch_20.pth
```