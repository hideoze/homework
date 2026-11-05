# 实验一

两题独立运行，各自在自己的 `results/` 目录保存 CSV、PDF 和图表。聚类仅保留花瓣长度、宽度双特征标准化方案，主展示 K-means k=3 与 DBSCAN，同时记录轮廓系数选出的 K-means 参数。

```bash
cd pagerank
python3 -m pip install numpy scipy matplotlib
python3 main.py
cd ../clustering
python3 -m pip install -r requirements.txt
python3 main.py
```

- [第 1 题：PageRank](pagerank/main.py)
- [第 2 题：聚类报告](clustering/results/report.pdf)
- 参考课件：[1.5 Clustering.pdf](<1.5 Clustering.pdf>)，第 2 题使用其中 1.5-1、1.5-3、1.5-5 节。

`submission.zip` 收录两题代码、数据、说明和非 PDF 结果。两题的 PDF 报告分别位于各自 `results/report.pdf`，不放入压缩包。
