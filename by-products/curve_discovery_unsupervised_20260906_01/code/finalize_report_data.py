from run_experiments import *
from sklearn.metrics import silhouette_score
import html,re

def markdown_html(text):
    """Small renderer for this report's headings, paragraphs, tables, lists and code."""
    def inline(s):
        s=html.escape(s)
        s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
        s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
        return re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    lines=text.splitlines();out=[];i=0
    while i<len(lines):
        s=lines[i]
        if s.startswith('```'):
            body=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):body.append(lines[i]);i+=1
            out.append('<pre>'+html.escape('\n'.join(body))+'</pre>')
        elif s.startswith('#'):
            n=len(s)-len(s.lstrip('#'));out.append(f'<h{n}>'+inline(s[n:].strip())+f'</h{n}>')
        elif s.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=lines[i].strip('|').split('|')
                if not all(re.fullmatch(r'[\s:\-]+',c) for c in cells):table.append(cells)
                i+=1
            out.append('<div class="table"><table>'+''.join('<tr>'+''.join(('<th>' if j==0 else '<td>')+inline(c.strip())+('</th>' if j==0 else '</td>') for c in row)+'</tr>' for j,row in enumerate(table))+'</table></div>');continue
        elif s.startswith('- '):
            out.append('<ul>')
            while i<len(lines) and lines[i].startswith('- '):out.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            out.append('</ul>');continue
        elif s.strip():out.append('<p>'+inline(s)+'</p>')
        i+=1
    return '<!doctype html><html lang="zh"><meta charset="utf-8"><title>无平滑曲线探索报告</title><style>body{font:16px/1.8 system-ui;max-width:1150px;margin:40px auto;padding:0 25px;color:#172033}h1{font-size:29px}h2{margin-top:40px}a{color:#2563eb}table{border-collapse:collapse;min-width:800px}td,th{padding:9px 13px;border:1px solid #dbe2ea;text-align:left}th{background:#eff6ff}.table{overflow:auto}code,pre{background:#f1f5f9}pre{padding:18px;overflow:auto}li{margin:6px 0}</style><a href="index.html">返回交互查看器</a>'+''.join(out)+'</html>'

def main():
    X=np.load(ROOT/'data/representation_minmax.npz')['X'];sel=json.loads((ROOT/'experiments/selection.json').read_text());rows=[]
    for norm,s in sel.items():
        labels=np.load(ROOT/f'experiments/models_{norm}.npz')['labels']
        centers=np.array([X[labels==j].mean(axis=0) for j in range(s['selected_k'])])
        rows.append({'method':f'functional_{norm}','selected_k':s['selected_k'],
           'seed_ari_mean':s['seed_ari_mean'],'silhouette_common_minmax_l2':float(silhouette_score(X,labels)),
           'assigned_common_l2_centroid_error':float(np.linalg.norm(X-centers[labels],axis=1).mean()),
           'own_qe':s['qe'],'own_qe_comparable_across_methods':False})
    elastic=json.loads((ROOT/'experiments/elastic_selection.json').read_text())
    for name,s in elastic.items():
        r=s['selection_row'];rows.append({'method':name,'selected_k':s['selected_k'],'seed_ari_mean':s['seed_ari_mean'],
            'silhouette_common_minmax_l2':r['silhouette_l2'],'assigned_common_l2_centroid_error':r['assigned_l2_centroid_error'],
            'own_qe':r['qe_elastic'],'own_qe_comparable_across_methods':False})
    pd.DataFrame(rows).to_csv(ROOT/'method_comparison.csv',index=False)
    counts={p.name:len(pd.read_csv(p)) for p in (ROOT/'experiments').glob('*sweep.csv')}
    counts['screening.csv']=len(pd.read_csv(ROOT/'experiments/screening.csv'))
    counts['group_bootstrap_training_runs']=20*16
    dump(ROOT/'experiments/training_counts.json',{'counts':counts,'total':sum(counts.values())})
    p=ROOT/'REPORT_CN.md';report=p.read_text()
    for row in rows:
        if row['method']=='functional_maxabs':report=report.replace('| MaxAbs SOM | 5 | 0.825 | 见比较 CSV |',f'| MaxAbs SOM | 5 | 0.825 | {row["silhouette_common_minmax_l2"]:.3f} |')
        if row['method']=='functional_zscore':report=report.replace('| z-score SOM | 5 | 0.700 | 见比较 CSV |',f'| z-score SOM | 5 | 0.700 | {row["silhouette_common_minmax_l2"]:.3f} |')
    if '浏览器实测状态' not in report:
        report+='\n\n**浏览器实测状态：** 此会话没有可连接的浏览器，无法完成真实浏览器交互实测；离线 HTML 已检查语法、内嵌数据与控件逻辑，静态图已人工目视核对。所有交互页数据均内嵌，不依赖网络。若查看器受本机浏览器限制，可直接使用 PNG 图库和逐曲线 CSV。\n'
    p.write_text(report);(ROOT/'report.html').write_text(markdown_html(report))
    index=ROOT/'index.html';s=index.read_text();s=s.replace('href="REPORT_CN.md"','href="report.html"')
    target='<p><a href="figures/00_four_target_examples_minmax.png">四种目标形态候选图</a> · <a href="target_shape_examples.csv">示例来源</a> · <a href="figures/local_parent_4.png">回升子群</a> · <a href="data/quality_flags.csv">数据质量标记</a></p>'
    if '四种目标形态候选图' not in s:s=s.replace('<label>分组',target+'<label>分组')
    note='<p class="note"><strong>当前判据支持4个主簇（46/23/89/60条），并不等同于四种IFO形态各自成为独立簇。</strong> 图片140的两条曲线记录了异常大的时间尺度，图片208有负输出且缺少单位；均保留，详见质量标记。</p>'
    if '当前判据支持4个主簇' not in s:s=s.replace('<label>分组',note+'<label>分组')
    index.write_text(s)
    # Exhaustive delivery consistency checks.
    inventory=pd.read_csv(ROOT/'data/curve_inventory.csv');a=pd.read_csv(ROOT/'cluster_assignments.csv')
    assert len(a)==218 and a.curve_id.is_unique and set(a.curve_id)==set(inventory.curve_id)
    assert len(list((ROOT/'data/transformed_curves').glob('*.csv')))==218
    assert sum(len(pd.read_csv(f)) for f in (ROOT/'data/transformed_curves').glob('*.csv'))==9674
    assert len(list((ROOT/'figures/all_curves').glob('page_*.png')))==19
    verification={'n_curves':218,'n_original_points':9674,'n_gallery_pages':19,'training_runs':sum(counts.values()),
        'standalone_html':True,'browser_tested':False,'numerical_verification':'experiments/verification.json'}
    dump(ROOT/'experiments/delivery_checks.json',verification)
    print('COUNTS',counts,'total',sum(counts.values()));print('COMPARISON',rows)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
