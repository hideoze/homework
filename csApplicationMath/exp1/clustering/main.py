from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from html import escape
import pymupdf
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'


def save_csv(name, header, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def cluster_metrics(x, truth, labels):
    keep = labels != -1
    count = len(set(labels[keep]))
    score = silhouette_score(x[keep], labels[keep]) if 1 < count < keep.sum() else np.nan
    ch = calinski_harabasz_score(x[keep], labels[keep]) if 1 < count < keep.sum() else np.nan
    dbi = davies_bouldin_score(x[keep], labels[keep]) if 1 < count < keep.sum() else np.nan
    return score, ch, dbi, adjusted_rand_score(truth, labels), normalized_mutual_info_score(truth, labels, average_method='arithmetic'), count, int((~keep).sum())


def main():
    OUT.mkdir(exist_ok=True)
    iris = load_iris()
    x = StandardScaler().fit_transform(iris.data[:, 2:])
    header = ['model', 'k_or_eps', 'min_samples', 'silhouette', 'CH', 'DBI', 'ARI', 'NMI', 'clusters', 'noise']
    candidates = []
    for k in range(2, 11):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)
        candidates.append((('kmeans', k, '', *cluster_metrics(x, iris.target, labels)), labels))
    radii = sorted(set(np.round(np.arange(0.2, 1.51, 0.1), 2)) | set(np.round(np.arange(0.6, 1.001, 0.02), 2)))
    for eps in radii:
        for minimum in [3, 5, 10]:
            labels = DBSCAN(eps=float(eps), min_samples=minimum).fit_predict(x)
            candidates.append((('dbscan', float(eps), minimum, *cluster_metrics(x, iris.target, labels)), labels))
    km_opt = max((item for item in candidates if item[0][0] == 'kmeans'), key=lambda item: item[0][3])
    km = next(item for item in candidates if item[0][0] == 'kmeans' and item[0][1] == 3)
    db = max((item for item in candidates if item[0][0] == 'dbscan' and np.isfinite(item[0][3]) and item[0][-1] <= 15), key=lambda item: item[0][3])
    save_csv('clustering_parameters.csv', header, [row for row, _ in candidates])
    save_csv('clustering_best.csv', ['selection', *header],
             [('known_k3', *km[0]), ('silhouette_noise_le_10pct', *db[0]), ('silhouette_kmeans', *km_opt[0])])
    save_csv('clustering_labels.csv', ['sample_id', 'true_label', 'kmeans_k3', 'dbscan', 'kmeans_silhouette'],
             zip(range(150), iris.target, km[1], db[1], km_opt[1]))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, sharex=True, sharey=True)
    for ax, labels, title in zip(axes, [iris.target, km[1], db[1]],
                                 ['True species', 'K-means: k=3', f'DBSCAN: eps={db[0][1]}, min_samples={db[0][2]}']):
        for group in np.unique(labels):
            keep = labels == group
            name = 'Noise' if group == -1 else (iris.target_names[group] if title == 'True species' else f'Cluster {group}')
            ax.scatter(*x[keep].T, color='gray' if group == -1 else plt.get_cmap('tab10')(int(group)),
                       marker='x' if group == -1 else 'o', s=25, label=name)
        ax.set(title=title, xlabel='Standardized petal length', ylabel='Standardized petal width')
        ax.legend(fontsize=7)
    fig.savefig(OUT / 'clustering.png', dpi=180)
    plt.close(fig)
    sections = [
        ('1）参数选择',
         '本实验使用 Iris 数据集，共150个样本，分为三个物种，每个物种50个样本。选取花瓣长度和花瓣宽度作为聚类特征，'
         '先对两列数据进行标准化，再分别运行 K-means 和 DBSCAN。\n\n'
         'K-means 的 k 在2到10之间取值，每组参数运行10次初始化，随机种子设为0。'
         f'其中 k={km_opt[0][1]} 的轮廓系数最高，为 {km_opt[0][3]:.4f}。'
         '不过，k=2会把两个较接近的物种合并。考虑到数据集有三个物种，后面的聚类图采用 k=3，'
         '同时在评价表中列出 k=2 的结果作比较。\n\n'
         'DBSCAN 的 ε 在0.2到1.5之间按0.1搜索，并在0.6到1.0之间补充步长为0.02的取值；'
         'min_samples 分别取3、5和10。为了避免大量样本被当作噪声，要求噪声比例不超过10%，'
         '再从符合条件的参数中选择轮廓系数最高的一组。\n\n'
         f'最终选出的参数为 ε={db[0][1]}、min_samples={db[0][2]}，得到{db[0][-2]}个簇和{db[0][-1]}个噪声点。'
         f'本次共比较{len(candidates)}组参数，详细结果见 clustering_parameters.csv。'
         '真实物种标签只用于计算 ARI 和 NMI，不参与参数评分。'),
        ('2）聚类结果可视化',
         '下图从左到右分别是真实物种、K-means（k=3）和 DBSCAN 的结果。'
         '横轴为标准化后的花瓣长度，纵轴为标准化后的花瓣宽度。由于只使用两个特征，可以直接画图，不需要再降维。'
         '灰色叉号表示噪声点，各图中的簇编号不一定与物种编号相同。\n\n'
         '从图中可以看到，setosa 与另外两个物种分得比较开，而 versicolor 和 virginica 有部分重叠。'
         'K-means 在 k=3 时能将它们大致分开，错误主要集中在两类交界处。'
         'DBSCAN 则把这两个相连的点群合并成一个簇，所以最后只有两个簇。'),
        ('3）评价指标与结果分析',
         '按照课件1.5-5节，使用 CH、轮廓系数、DBI、ARI 和 NMI 进行比较。'
         'CH 和轮廓系数越大越好，DBI 越小越好，这三项主要反映簇内是否紧密、簇间是否分离。'
         'ARI 和 NMI 越接近1，说明聚类划分与真实物种越一致。NMI 采用算术平均熵归一化。\n\n'
         f'K-means（k=3）的 ARI 为 {km[0][6]:.4f}，NMI 为 {km[0][7]:.4f}，均高于 DBSCAN，'
         '说明它的结果更接近真实物种划分。相比 k=2，k=3 的轮廓系数有所下降，但 ARI 和 NMI 更高。'
         '这也说明，只按轮廓系数选参数，不一定能得到最接近真实类别的结果。\n\n'
         f'DBSCAN 的轮廓系数为 {db[0][3]:.4f}，DBI 为 {db[0][5]:.4f}，这两项优于 K-means。'
         '但它把两个物种合并了，因此 ARI 和 NMI 较低。计算 DBSCAN 的 CH、轮廓系数和 DBI 时排除了噪声点；'
         '计算 ARI 和 NMI 时保留全部150个样本，并将噪声标签-1作为一组，所以不能只根据内部指标判断哪个模型更好。\n\n'
         '综合图像和五项指标，本次实验中 K-means（k=3）更适合区分三个物种。'
         'DBSCAN 能找出较独立的点群和噪声，但对密度相连的两个物种区分较差。'
         '因此，这组数据上采用 K-means（k=3）的结果更符合按物种划分的目标。')
    ]
    table_header = ['方案', 'CH ↑', '轮廓 ↑', 'DBI ↓', 'ARI ↑', 'NMI ↑', '噪声']
    table_rows = [[name, *[f'{row[i]:.4f}' for i in [4,3,5,6,7]], str(row[-1])]
                  for name, (row, _) in zip(['K-means k=3', 'DBSCAN', 'K-means k=2'], [km,db,km_opt])]
    table = '<table><tr>'+''.join('<th>'+escape(v)+'</th>' for v in table_header)+'</tr>'
    table += ''.join('<tr>'+''.join('<td>'+escape(v)+'</td>' for v in row)+'</tr>' for row in table_rows)+'</table>'
    css = ('body {font-family:sans-serif;font-size:10pt;}'
           'h2 {font-size:13pt;margin:0 0 8pt;}'
           'h3 {font-size:11pt;margin:8pt 0 5pt;}'
           'p {margin:0 0 6pt;line-height:1.3;}'
           'table {border-collapse:collapse;width:100%;font-size:8pt;margin-bottom:8pt;}'
           'td,th {border:1px solid #888;padding:3pt;}')
    html = '<h2>实验二：聚类分析</h2><p>代码文件：clustering/main.py</p>'
    for i, (title, text) in enumerate(sections):
        html += '<h3>' + escape(title) + '</h3>' + (table if i == 2 else '')
        html += ''.join('<p>' + escape(p) + '</p>' for p in text.split('\n\n'))
        if i == 1:
            html += '<p style="text-align:center"><img src="clustering.png" width="480" height="160"></p>'
    with pymupdf.open() as doc:
        page = doc.new_page(width=595, height=842)
        spare, scale = page.insert_htmlbox(
            pymupdf.Rect(38, 28, 557, 802), html, css=css,
            archive=pymupdf.Archive(str(OUT)))
        assert spare >= 0 and scale > 0.95
        page.insert_text((285, 820), '1', fontsize=9)
        doc.save(OUT / 'report.pdf')
    summary='\n\n'.join(title+'\n\n'+text for title,text in sections)
    summary+='\n\n'+'\n'.join(' | '.join(row) for row in [table_header,*table_rows])+'\n'
    (OUT/'summary.txt').write_text(summary)
    print(summary)


if __name__ == '__main__':
    main()
