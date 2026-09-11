#!/usr/bin/env python3
"""Freeze audited Que baseline evidence, then prune explicitly scoped old weights.

No training, no process signals, no metric selection, no weight deserialization.
Default mode only prepares and verifies retention; --execute also prunes.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tarfile
import tempfile

MODELS = ('gru', 'lstm', 'msnet', 'mscmnet_m', 'mscmnet_wm', 'mscmnet_w')
HISTORY_ROOTS = tuple('results/' + name for name in (
    'que_reproduction_matrix_20260831', 'que_attention_diagnostics_20260901',
    'que_optimizer_diagnostics_20260901', 'que_cam_layout_diagnostics_20260901',
    'que_selected_joint_baselines_20260901', 'que_correction_calibration_diagnostics_20260902',
    'que_final_reproduction_20260903', 'que_complete_reproduction_20260903',
    'que_targeted_reproduction_20260904', 'que_protocol_audit_20260905',
    'que_comprehensive_reconstruction_20260906', 'que_comprehensive_reconstruction_shared_20260908',
    'que_recurrent_focus_20260910', 'msnet_cam_table3_smoke', 'msnet_cam_table3_zscore',
    'msnet_cam_table3_minmax', 'temporal_baselines', 'temporal_baselines_smoke'))
HISTORY_LOG_ROOTS = tuple('logs/' + Path(x).name for x in HISTORY_ROOTS) + tuple('logs/' + x for x in (
    'que_attention_diagnostics_launcher.log','que_optimizer_diagnostics_launcher.log',
    'que_cam_layout_diagnostics_launcher.log','que_selected_joint_baselines_launcher.log',
    'que_correction_matched_resume.log','que_correction_calibration_diagnostics_launcher.log',
    'que_final_reproduction_launcher.log','que_complete_reproduction_launcher.log',
    'que_targeted_reproduction_launcher.log','que_protocol_audit_launcher.log',
    'que_comprehensive_reconstruction_launcher.log','que_shared_gpu7_launcher.log',
    'que_recurrent_focus_launcher.log'))
# These are the six temporal families only: no graph models, arbitrary .pt files or source data.
CHECKPOINT = re.compile(r'checkpoint_(?:gru|lstm)_dma_[A-J]\.pt$|checkpoint_(?:msnet|mscmnet_m|mscmnet_wm|mscmnet_w)\.pt$')
ACTIVE = {'run_que_comprehensive_reconstruction.py', 'run_que_total_focus.py',
          'run_que_protocol_audit.py', 'train_temporal_baselines.py', 'que_shared_gpu_runtime.py'}


def now(): return datetime.now(timezone.utc).isoformat()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def relative(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or str(p) != str(value):
        raise ValueError('Unsafe relative path: ' + str(value))
    return p


def safe(path, *, directory=False):
    path = Path(path).absolute()
    for parent in (*reversed(path.parents), path):
        if parent.is_symlink(): raise ValueError('Symlink is outside cleanup scope: ' + str(parent))
    st = path.stat()
    if directory and not stat.S_ISDIR(st.st_mode): raise ValueError('Not a directory: ' + str(path))
    if not directory and not stat.S_ISREG(st.st_mode): raise ValueError('Not a regular file: ' + str(path))
    return st


def file_sha(path):
    safe(path); h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def inventory(root):
    root = Path(root); safe(root, directory=True); result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('Symlink in retained evidence: ' + str(path))
        if path.is_dir(): continue
        result[str(path.relative_to(root))] = file_sha(path)
    return result


def check_inventory(root, expected):
    if inventory(root) != expected: raise ValueError('Retained file inventory/hash differs: ' + str(root))


def verify_tar(path, expected):
    actual = {}
    with tarfile.open(path, 'r:gz') as tar:
        for m in tar:
            relative(m.name)
            if not m.isfile() or m.name in actual: raise ValueError('Unsafe/duplicate archive member: ' + m.name)
            f = tar.extractfile(m); h = hashlib.sha256()
            for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
            actual[m.name] = h.hexdigest()
    if actual != expected: raise ValueError('Archive content hash differs: ' + str(path))


def create_tar(archive, items):
    with tarfile.open(archive, 'w:gz') as tar:
        for name, path in sorted(items.items()):
            relative(name); safe(path)
            tar.add(path, arcname=name, recursive=False)


def assert_idle(project):
    """Inspect only this user's relevant trainers; never signal any process."""
    project = Path(project).absolute()
    focused = project / 'results/que_recurrent_focus_20260910/campaign_status.json'
    if focused.exists():
        status = read(focused)
        if status.get('status') != 'completed_search' or status.get('active_case'):
            raise RuntimeError('Focused search has not finished; no files were removed')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
        try:
            if proc.stat().st_uid != os.getuid(): continue
            words = (proc / 'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError: continue
        except PermissionError: continue  # Unrelated protected processes are not targets.
        argv = [w.decode(errors='replace') for w in words if w]
        relevant = any(Path(w).name in ACTIVE or (Path(w).name.startswith('run_que_') and w.endswith('.sh')) for w in argv)
        if not relevant: continue
        try: cwd = (proc / 'cwd').resolve(strict=True)
        except FileNotFoundError: continue
        except PermissionError as exc: raise RuntimeError('Cannot identify a potentially relevant trainer: ' + proc.name) from exc
        belongs = cwd == project or cwd.is_relative_to(project) or any(str(project) in w for w in argv)
        belongs = belongs or (cwd.is_relative_to(project.parent / 'que_total_focus_tools'))
        if belongs: raise RuntimeError('Project training process is still active: PID=' + proc.name)


def lock_file(stack, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    safe(path.parent, directory=True)
    if path.is_symlink(): raise ValueError('Symlink lock is not accepted')
    handle = stack.enter_context(path.open('a+'))
    try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc: raise RuntimeError('A project queue/cleanup still holds ' + str(path)) from exc


def check_metrics(path, selected):
    import numpy as np
    with np.load(path, allow_pickle=False) as z: a = {k: z[k] for k in z.files}
    if a['dma_letters'].tolist() != list('ABCDEFGHIJ'): raise ValueError('DMA ordering mismatch')
    origins = a['forecast_starts'].tolist()
    if len(origins) != 46 or len(set(origins)) != 46: raise ValueError('Evaluation origin mismatch')
    values = []
    for task, length in [('24h',24), ('168h',168)]:
        y, p = a['y_true_'+task], a['y_pred_'+task]
        if y.shape != (46,length,10) or p.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError('Invalid prediction arrays')
        y, p = y.astype('float64'), p.astype('float64')
        yt, pt = y.sum(2), p.sum(2); error = yt - pt
        if np.any(np.abs(yt) < 1e-12) or np.var(yt) <= 0: raise ValueError('Undefined total metrics')
        values.extend([float(np.abs(y-p).mean((0,1)).sum()), float(np.mean(np.abs(error/yt))),
                       float(np.sqrt(np.mean(error**2))), float(1-np.mean(error**2)/np.var(yt))])
    if not np.array_equal(a['y_true_24h'], a['y_true_168h'][:,:24]) or not np.allclose(a['y_pred_24h'],a['y_pred_168h'][:,:24],rtol=1e-5,atol=1e-5):
        raise ValueError('24h and 168h first day mismatch')
    if not np.allclose(values, selected['expected_metrics'], rtol=1e-11, atol=1e-11):
        raise ValueError('Selected prediction metrics differ from frozen comparison')
    return values


def scan_history(project, protected):
    metadata, planned = {}, []
    for name in HISTORY_ROOTS:
        root = project / relative(name)
        if not root.exists() and not root.is_symlink(): continue
        safe(root, directory=True)
        for path in sorted(root.rglob('*')):
            if path.is_symlink(): raise ValueError('Symlink in experiment history: '+str(path))
            if path.is_dir(): continue
            st = safe(path); rel = str(path.relative_to(project))
            if CHECKPOINT.fullmatch(path.name):
                if rel not in protected:
                    planned.append(dict(path=rel, sha256=file_sha(path), size=st.st_size,
                        device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns))
                continue
            # Existing archive packages remain in place; do not duplicate embedded weights.
            if path.suffix in {'.pt','.lock','.tmp','.pyc','.pyo'} or '__pycache__' in path.parts or path.name.endswith(('.tar.gz','.zip')):
                continue
            metadata[rel] = file_sha(path)
    for name in HISTORY_LOG_ROOTS:
        root=project/relative(name)
        if not root.exists() and not root.is_symlink():continue
        if root.is_symlink():raise ValueError('Symlink historical log path')
        paths=sorted(root.rglob('*')) if root.is_dir() else [root]
        for path in paths:
            if path.is_symlink():raise ValueError('Symlink historical log')
            if path.is_dir():continue
            if path.suffix in {'.lock','.tmp','.pyc','.pyo','.pt'} or path.name.endswith(('.tar.gz','.zip')):continue
            metadata[str(path.relative_to(project))]=file_sha(path)
    return metadata, planned


def validate_plan(plan, selection):
    if plan.get('signature') != digest({k:v for k,v in plan.items() if k!='signature'}): raise ValueError('Cleanup plan changed')
    allowed = [relative(x) for x in plan['allowed_roots']]
    if tuple(plan['allowed_roots']) != HISTORY_ROOTS: raise ValueError('Cleanup scope changed')
    protected = {str(relative(m['source_run_relative']) / relative(x)) for m in selection['models'].values() for x in m['artifact_sha256']}
    seen = set()
    for rec in plan['files']:
        p = relative(rec['path'])
        if rec['path'] in seen or rec['path'] in protected or not CHECKPOINT.fullmatch(p.name) or not any(p.is_relative_to(a) for a in allowed):
            raise ValueError('Unsafe or protected cleanup target: '+rec['path'])
        seen.add(rec['path'])


def verify_ready(destination, selection=None):
    destination=Path(destination); safe(destination,directory=True)
    report=read(destination/'retention_manifest.json')
    if report.get('signature') != digest({k:v for k,v in report.items() if k!='signature'}): raise ValueError('Retention manifest changed')
    if report.get('status')!='READY':raise ValueError('Retention is not READY')
    stored=read(destination/'selection.json')
    if digest(stored)!=report['selection_sha256'] or (selection is not None and stored!=selection): raise ValueError('Selected baseline definition changed')
    plan=read(destination/'cleanup_plan.json')
    if digest(plan)!=report['cleanup_plan_sha256']: raise ValueError('Cleanup plan changed after retention')
    if plan.get('selection_sha256')!=digest(stored) or plan.get('project_root')!=report.get('project_root'):
        raise ValueError('Cleanup plan binding differs from retention')
    validate_plan(plan,stored)
    # Hash every retained model and history backup before trusting READY.
    for name,expected in report['payload_sha256'].items():
        if file_sha(destination/relative(name))!=expected: raise ValueError('Retained evidence hash mismatch: '+name)
    verify_tar(destination/'best_models_complete.tar.gz',report['model_archive_members'])
    verify_tar(destination/'history_records.tar.gz',report['history_archive_members'])
    for model,item in stored['models'].items():
        check_inventory(destination/'models'/model,item['artifact_sha256'])
        check_metrics(destination/'models'/model/'predictions_common46.npz',item)
    return report


def _validate_selection(selection):
    if set(selection['models']) != set(MODELS) or selection.get('seed')!=20240604 or selection.get('mode')!='pooled':
        raise ValueError('Expected the six reviewed single-seed pooled baselines')
    for model,item in selection['models'].items():
        relative(item['source_run_relative'])
        if len(item['expected_metrics'])!=8 or len(item['paper_targets'])!=8: raise ValueError('Incomplete eight-metric selection')
        expected=10 if model in ('gru','lstm') else 1
        if len(item['checkpoints'])!=expected: raise ValueError('Missing complete model checkpoint set')
        for cp in item['checkpoints']:
            relative(cp)
            if cp not in item['artifact_sha256'] or not CHECKPOINT.fullmatch(Path(cp).name): raise ValueError('Invalid selected checkpoint')


def prepare(project,selection,destination):
    _validate_selection(selection)
    if destination.exists(): return verify_ready(destination,selection)
    protected={str(relative(m['source_run_relative'])/relative(x)) for m in selection['models'].values() for x in m['artifact_sha256']}
    print('核验六个推荐配置的24个权重文件、参数及预测结果。',flush=True)
    for item in selection['models'].values():
        if file_sha(project/relative(item['source_manifest_relative'])) != item['source_manifest_sha256']:
            raise ValueError('Selected source manifest changed')
        src=project/relative(item['source_run_relative']); check_inventory(src,item['artifact_sha256']);check_metrics(src/'predictions_common46.npz',item)
    print('建立历史记录和待清理权重清单；此时不删除文件。',flush=True)
    metadata,planned=scan_history(project,protected)
    plan=dict(version=1,project_root=str(project),selection_sha256=digest(selection),allowed_roots=list(HISTORY_ROOTS),
        created_utc=now(),files=planned,retains_all_prediction_arrays=True,retains_original_records=True,
        does_not_delete_archives=True,planned_bytes=sum(x['size'] for x in planned))
    plan['signature']=digest(plan);validate_plan(plan,selection)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.que_retention_',dir=destination.parent) as tmp:
        stage=Path(tmp)/'retained';stage.mkdir()
        write(stage/'selection.json',selection);write(stage/'cleanup_plan.json',plan)
        for model,item in selection['models'].items():
            src=project/relative(item['source_run_relative']);dst=stage/'models'/model
            for rel,expected in item['artifact_sha256'].items():
                source=src/relative(rel);target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True)
                if file_sha(source)!=expected: raise ValueError('Selected evidence changed during copy')
                shutil.copyfile(source,target)
                if file_sha(target)!=expected: raise ValueError('Retained copy hash mismatch')
            check_inventory(dst,item['artifact_sha256'])
        source_root=project/'results/que_recurrent_focus_20260910/joint_closeout/source_snapshot'
        for name,expected in selection.get('numerical_source_sha256',{}).items():
            source=source_root/relative(name)
            if file_sha(source)!=expected: raise ValueError('Numerical source snapshot changed: '+name)
            target=stage/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        # Preserve the archival and replay recipe itself.
        shutil.copyfile(__file__,stage/'finalize_que_paper_baselines.py')
        write(stage/'history_inventory.json',metadata)
        with (stage/'total_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f);w.writerow(['model','horizon','MAE','MAPE','RMSE','NSE','case'])
            for model,item in selection['models'].items():
                for i,task in enumerate(('24h','168h')):w.writerow([model,task,*item['expected_metrics'][4*i:4*i+4],item['case']])
        with (stage/'paper_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f);w.writerow(['model','horizon','metric','current','paper','difference','tolerance_ratio','within_tolerance','case'])
            for model,item in selection['models'].items():
                for i,(v,p) in enumerate(zip(item['expected_metrics'],item['paper_targets'])):
                    metric=('MAE','MAPE','RMSE','NSE')[i%4];ratio=abs(v-p)/(.01 if metric=='NSE' else .05*abs(p))
                    w.writerow([model,('24h','168h')[i//4],metric,v,p,v-p,ratio,ratio<=1,item['case']])
        (stage/'README_CN.md').write_text('# 六模型固定结果集\n\n模型、参数、归一化及预测保存在 models/；total_metrics.csv 可供实验表格整理。\n'
            '本次选择固定seed20240604，按公开论文测试目标选择最小最大容差偏差；这是回顾性数值重建，不能声称独立测试验证或完全恢复作者实现。\n'
            '当前37/48指标通过既定容差。MAE为DMA MAE之和，其余三项基于总需求pooled计算。\n'
            'history_records.tar.gz 保留所有清理范围内的原始参数、指标、预测、失败和过程记录。旧checkpoint的清理记录见cleanup_plan.json和cleanup_journal.jsonl；原历史PASS状态不作篡改。\n'
            '后续应从本目录的模型副本读取，旧目录在清理后不再保证完整checkpoint。源模型保留原字节，未独立加载重推理。\n')
        model_items={str(p.relative_to(stage)):p for p in stage.rglob('*') if p.is_file() and p.name!='history_records.tar.gz'}
        model_hashes={n:file_sha(p) for n,p in model_items.items()}
        create_tar(stage/'best_models_complete.tar.gz',model_items);verify_tar(stage/'best_models_complete.tar.gz',model_hashes)
        print(f'六个模型备份通过；备份历史记录 {len(metadata)} 个文件。',flush=True)
        create_tar(stage/'history_records.tar.gz',{n:project/n for n in metadata});verify_tar(stage/'history_records.tar.gz',metadata)
        # A concurrent producer or changed source must stop cleanup even if the copy was valid earlier.
        check_again,plan_again=scan_history(project,protected)
        if check_again!=metadata or plan_again!=planned: raise ValueError('Experiment history changed during retention; no cleanup allowed')
        for item in selection['models'].values():check_inventory(project/item['source_run_relative'],item['artifact_sha256'])
        assert_idle(project)
        report=dict(status='READY',created_utc=now(),project_root=str(project),selection_sha256=digest(selection),
            cleanup_plan_sha256=digest(plan),planned_files=len(planned),planned_bytes=plan['planned_bytes'],
            payload_sha256=inventory(stage),model_archive_members=model_hashes,history_archive_members=metadata,
            source_weights_retained=True,independent_test_validation=False)
        report['signature']=digest(report);write(stage/'retention_manifest.json',report)
        verify_ready(stage,selection)
        os.replace(stage,destination)
    print('保留包及历史备份已逐文件核验：'+str(destination),flush=True)
    return verify_ready(destination,selection)


def append_journal(path,record):
    with path.open('a') as f:
        f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())


def _checked_parent(project,path):
    """Walk with directory descriptors so no parent symlink can redirect unlink."""
    fd=os.open(project,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in relative(path).parts[:-1]:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
        return fd
    except BaseException:os.close(fd);raise


def remove_planned(project,record):
    fd=_checked_parent(project,record['path']);name=Path(record['path']).name
    try:
        st=os.stat(name,dir_fd=fd,follow_symlinks=False)
        actual=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        expected=(record['device'],record['inode'],record['size'],record['mtime_ns'])
        if not stat.S_ISREG(st.st_mode) or actual!=expected: raise ValueError('Cleanup target identity changed: '+record['path'])
        opened=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        with os.fdopen(opened,'rb') as f:
            h=hashlib.sha256()
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        if h.hexdigest()!=record['sha256']: raise ValueError('Cleanup target bytes changed: '+record['path'])
        st=os.stat(name,dir_fd=fd,follow_symlinks=False)
        if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=expected: raise ValueError('Cleanup target changed before unlink')
        os.unlink(name,dir_fd=fd)
    finally:os.close(fd)


def clean(project,selection,destination,report):
    plan=read(destination/'cleanup_plan.json');validate_plan(plan,selection)
    if plan['project_root']!=str(project): raise ValueError('Cleanup project changed')
    journal_path=destination/'cleanup_journal.jsonl';events=[]
    if journal_path.exists():
        safe(journal_path)
        for line in journal_path.read_text().splitlines():
            if line.strip():events.append(json.loads(line))
    known={r['path']:r for r in plan['files']};last={}
    for event in events:
        if event.get('path') not in known or event.get('sha256')!=known[event['path']]['sha256'] or event.get('state') not in ('PENDING','DELETED'):
            raise ValueError('Cleanup journal differs from immutable plan')
        last[event['path']]=event['state']
    # Refuse stale preservation before any unlink, including a later --execute resume.
    for name,expected in report['history_archive_members'].items():
        if file_sha(project/relative(name))!=expected:
            raise ValueError('Retained historical evidence changed before cleanup: '+name)
    # Validate every remaining target before the first new deletion.
    for rec in plan['files']:
        p=project/rec['path']
        if not p.exists() and not p.is_symlink():
            if rec['path'] not in last: raise ValueError('Target missing before recorded cleanup: '+rec['path'])
            continue
        if last.get(rec['path'])=='DELETED':raise ValueError('Previously deleted target was recreated; refusing to delete it')
        st=safe(p)
        if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=(rec['device'],rec['inode'],rec['size'],rec['mtime_ns']) or file_sha(p)!=rec['sha256']:
            raise ValueError('Planned target changed: '+rec['path'])
    assert_idle(project)
    for number,rec in enumerate(plan['files'],1):
        p=project/rec['path']
        if not p.exists():
            if last.get(rec['path'])!='DELETED':append_journal(journal_path,dict(path=rec['path'],sha256=rec['sha256'],state='DELETED',recovered_pending=True,time=now()))
            continue
        append_journal(journal_path,dict(path=rec['path'],sha256=rec['sha256'],state='PENDING',time=now()))
        remove_planned(project,rec)
        append_journal(journal_path,dict(path=rec['path'],sha256=rec['sha256'],state='DELETED',time=now()))
        if number%50==0:print(f'已清理未选权重：{number}/{len(plan["files"])}',flush=True)
    # Metadata/results remain in their original locations and in a verified archive.
    for name,expected in report['history_archive_members'].items():
        if file_sha(project/name)!=expected: raise ValueError('Retained historical evidence changed: '+name)
    verify_ready(destination,selection)
    status=dict(status='CLEANED',updated_utc=now(),planned_files=len(plan['files']),deleted_files=len(plan['files']),
        deleted_bytes=sum(x['size'] for x in plan['files']),destination=str(destination),
        noncheckpoint_files_deleted=0,new_unplanned_files_deleted=0)
    write(destination/'cleanup_status.json',status);return status


def finalize(project,selection,destination,execute=False):
    project=Path(project).absolute();destination=Path(destination).absolute();safe(project,directory=True)
    expected=project/relative(selection['output_relative'])
    if destination!=expected or any(destination==project/x or destination.is_relative_to(project/x) for x in HISTORY_ROOTS):
        raise ValueError('Retention destination must be the separate selected output directory')
    # Never follow an existing destination symlink, including during resume.
    for p in (*destination.parents,destination):
        if p.is_symlink():raise ValueError('Symlink retention destination')
    with ExitStack() as stack:
        lock_file(stack,project/'logs/que_paper_closeout.lock')
        lock_file(stack,project/'logs/que_gpu_7.lock')
        lock_file(stack,project/'results/que_recurrent_focus_20260910/campaign.lock')
        assert_idle(project)
        report=prepare(project,selection,destination)
        print(f'待清理未选权重：{report["planned_files"]} 个，{report["planned_bytes"]/1024**3:.3f} GiB；参数和结果保留。',flush=True)
        if execute:return clean(project,selection,destination,report)
        return {**report,'deleted_files':0,'deleted_bytes':0}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path.home()/'projects/STaR-GNN-BWDF')
    p.add_argument('--selection',type=Path,default=Path(__file__).resolve().parents[2]/'configs/evaluation/que_selected_total8_20260911.json')
    p.add_argument('--execute',action='store_true');p.add_argument('--status',action='store_true')
    args=p.parse_args(argv);selection=read(args.selection);destination=args.project_root/relative(selection['output_relative'])
    try:
        if args.status:
            status=destination/'cleanup_status.json'
            print(json.dumps(read(status) if status.exists() else {'status':'READY' if (destination/'retention_manifest.json').exists() else 'not_prepared','destination':str(destination)},ensure_ascii=False,indent=2));return 0
        report=finalize(args.project_root,selection,destination,execute=args.execute)
        print(json.dumps({k:report[k] for k in ('status','planned_files','deleted_files','deleted_bytes') if k in report},ensure_ascii=False,indent=2))
        print('固定模型和论文比较表：'+str(destination),flush=True);return 0
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        print('发生错误时不会扩大清理范围；已完成的模型备份及清理日志保留。',file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
