### Behaviors


- [x] 全局联合侦察

- [x] 要地防护

- [x] 聚集

- [x] 大规模编队（变阵）

- [x] 隐蔽隐藏

- [x] 长机僚机3d

- [x] 一字型

- [x] 巡逻

- [x] 钻洞（3D）16uav

- [x] 局部联合侦察

- [x] 穿林 

	
### Result

- CNN 5Rounds

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      2000
           1       1.00      0.99      1.00      2000
           2       1.00      1.00      1.00      2000
           3       1.00      1.00      1.00      2000
           4       0.92      0.67      0.77      2000
           5       1.00      1.00      1.00      2000
           6       0.99      0.99      0.99      2000
           7       1.00      1.00      1.00      2000
           8       1.00      0.99      1.00      2000
           9       0.99      0.99      0.99      2000
          10       0.74      0.95      0.83      2000

    accuracy                           0.96     22000
   macro avg       0.97      0.96      0.96     22000
weighted avg       0.97      0.96      0.96     22000
```

```
[[2000    0    0    0    0    0    0    0    0    0    0]
 [   2 1990    0    1    0    0    1    0    0    0    6]
 [   0    0 1999    0    1    0    0    0    0    0    0]
 [   0    1    0 1997    0    0    0    0    0    0    2]
 [   0    1    0    3 1331    0   13    0    0    9  643]
 [   0    0    0    0    0 1999    0    1    0    0    0]
 [   0    0    0    0    9    0 1990    0    0    0    1]
 [   0    0    1    0    1    0    0 1995    0    3    0]
 [   0    0    0    0    2    0    0    0 1988    0   10]
 [   0    0    0    0    6    3    0    2    0 1989    0]
 [   0    3    0    1   97    0    0    0    1    0 1898]]
```

 - LSTM 99 rounds, 15 steps

```
              precision    recall  f1-score   support

           0       0.76      0.77      0.77      2000
           1       1.00      0.99      1.00      2000
           2       0.97      0.99      0.98      2000
           3       0.99      1.00      0.99      2000
           4       0.81      0.78      0.80      2000
           5       1.00      1.00      1.00      2000
           6       1.00      1.00      1.00      2000
           7       0.96      0.93      0.94      2000
           8       1.00      1.00      1.00      2000
           9       0.73      0.74      0.74      2000
          10       0.82      0.83      0.82      2000

    accuracy                           0.91     22000
   macro avg       0.91      0.91      0.91     22000
weighted avg       0.91      0.91      0.91     22000
```

```
[[1549    3    2    5    5    0    0   21    0  411    4]
 [   5 1989    0    3    1    0    0    0    0    1    1]
 [   1    0 1984    1    9    0    1    0    0    1    3]
 [   4    0    1 1993    0    0    0    0    0    2    0]
 [   9    0   43    2 1564    2    1    2    0   20  357]
 [   0    0    0    0    0 1996    0    0    4    0    0]
 [   0    0    0    0    0    0 1999    0    0    0    1]
 [  25    0    0    1    5    0    0 1855    0  114    0]
 [   0    0    0    0    0    4    0    2 1994    0    0]
 [ 429    1    4    7   18    0    1   56    0 1484    0]
 [  14    1    6    1  323    0    1    2    0    1 1651]]
```

### Mean - Std

```
hideds [27.7553433  14.32217516  0.        ] [15.44525121 12.44175005  1.        ]
flockingds [-0.00911305 -0.07189604  0.        ] [0.13288371 0.2279316  1.        ]
lineds [26.46675763 26.82483649  0.        ] [6.36458332 7.46127806 1.        ]
pass3d [8.07585196 4.89015749 1.07418603] [3.85663151 1.94566411 0.34342566]
leaderfollowers_3d [156.08249034 147.74549953 148.35972137] [69.03518208 66.09661156 67.50345974]
treesds2 [27.2549865  16.41504084  0.        ] [15.03331502 10.08243792  1.        ]
patrol [298.85699612 323.41394958   0.        ] [181.8650429  186.08196638   1.        ]
defenseds2 [11.13655346 28.98705326  0.        ] [5.4021141  6.45441055 1.        ]
cppds [19.77328481 18.98573191  0.        ] [11.23528616 11.0552819   1.        ]
poi [0.53594744 0.48621241 0.        ] [0.21121522 0.20142002 1.        ]
formation [32.34946692 36.71211851  0.        ] [ 9.633409   12.54327206  1.        ]
```

