input data：
TIF files, each TIF file has dimensions of A×B, with a total of 25 files in this paper.

Execution Steps:
1.Contrastive network.py
2.Organize the data to obtain the model parameters with the closest and farthest distances.
3.output image of Contrastive_network.py
4.3guiyihua.py(Max-min normalization)
5.AUCHE.Py（Average the normalization results representing the closest and farthest）
6.Pu-baggingDT.py
7.Precision_evaluation.py（Average operation: Results from Pu-baggingDT and AUCHE.py，and precision evaluation）
