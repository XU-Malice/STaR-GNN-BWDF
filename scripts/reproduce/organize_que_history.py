#!/usr/bin/env python3
"""Consolidate completed Que experiment directories without deleting their contents.

Requires the verified paper-baseline retention and CLEANED receipt. The fixed
history allowlist is moved intact on the same filesystem. Old absolute paths
remain historical evidence; path_mapping.tsv records their new locations.
Default previews history and saves its immutable plan; --execute performs resumable moves.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import errno
import json
import os
import platform
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalize_que_paper_baselines as retention

CANONICAL_RELATIVE = 'results/que_paper_baselines_20260911'
ARCHIVE_RELATIVE = 'results/archive/que_reproduction'
HISTORY_ROOTS = retention.HISTORY_ROOTS + ('results/que_comprehensive_reconstruction_20260907',)
PLAN_FILE = 'organization_plan.json'
JOURNAL_FILE = 'organization_journal.jsonl'
STATUS_FILE = 'organization_status.json'
FOCUS_NAME = 'que_recurrent_focus_20260910'


def exists(path):
    return Path(path).exists() or Path(path).is_symlink()


def safe_parents(path):
    path = Path(path).absolute()
    for parent in (*reversed(path.parents), path):
        if parent.is_symlink():
            raise ValueError('Symlink in organization path: ' + str(parent))
        if parent.exists() and not parent.is_dir():
            raise ValueError('Organization path is not a directory: ' + str(parent))


def safe_mkdir(path):
    safe_parents(path)
    Path(path).mkdir(parents=True, exist_ok=True)
    retention.safe(path, directory=True)


def identity(st):
    return dict(device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns)


def inventory(root):
    """Include empty directories and opaque files, including leftover archives."""
    root = Path(root)
    dirs = {'.': identity(retention.safe(root, directory=True))}
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink in history; no moves permitted: ' + str(path))
        name = str(path.relative_to(root))
        if path.is_dir():
            dirs[name] = identity(retention.safe(path, directory=True))
        else:
            st = retention.safe(path)
            checksum = retention.file_sha(path)
            after = retention.safe(path)
            if identity(st) != identity(after) or st.st_size != after.st_size:
                raise ValueError('History changed while hashing: ' + str(path))
            files[name] = dict(identity(st), size=st.st_size, sha256=checksum)
    return dict(dirs=dirs, files=files)


def check_tree(root, tree):
    if inventory(root) != tree:
        raise ValueError('History inventory or file hash changed: ' + str(root))


def location(project, source):
    return project / source, project / ARCHIVE_RELATIVE / Path(source).name


def build_plan(project, canonical_report):
    roots = []
    absent = []
    archive = project / ARCHIVE_RELATIVE
    safe_parents(archive)
    for name in HISTORY_ROOTS:
        retention.relative(name)
        source, dest = location(project, name)
        if exists(dest):
            raise ValueError('Archive destination already exists without a recorded plan: ' + str(dest))
        if not exists(source):
            absent.append(name)
            continue
        tree = inventory(source)
        if tree['dirs']['.']['device'] != project.stat().st_dev:
            raise ValueError('History must be on the project filesystem: ' + name)
        roots.append(dict(source=name, destination=str(dest.relative_to(project)), tree=tree))
    plan = dict(version=1, project_root=str(project), canonical_relative=CANONICAL_RELATIVE,
                archive_relative=ARCHIVE_RELATIVE, allowed_roots=list(HISTORY_ROOTS),
                cleanup_status_sha256=retention.file_sha(project / CANONICAL_RELATIVE / 'cleanup_status.json'),
                retention_manifest_sha256=retention.file_sha(project / CANONICAL_RELATIVE / 'retention_manifest.json'),
                retention_signature=canonical_report['signature'], created_utc=retention.now(),
                roots=roots, absent_roots=absent)
    plan['signature'] = retention.digest(plan)
    return plan


def validate_plan(plan, project):
    unsigned = {k: v for k, v in plan.items() if k != 'signature'}
    if plan.get('signature') != retention.digest(unsigned):
        raise ValueError('Organization plan signature changed')
    if (plan.get('version') != 1 or plan.get('project_root') != str(project)
            or plan.get('canonical_relative') != CANONICAL_RELATIVE
            or plan.get('archive_relative') != ARCHIVE_RELATIVE
            or plan.get('allowed_roots') != list(HISTORY_ROOTS)):
        raise ValueError('Organization plan scope or project binding changed')
    names = []
    for item in plan['roots']:
        source = str(retention.relative(item['source']))
        dest = str(retention.relative(item['destination']))
        if source not in HISTORY_ROOTS or dest != str(Path(ARCHIVE_RELATIVE) / Path(source).name):
            raise ValueError('Unapproved history path in organization plan')
        names.append(source)
        tree = item['tree']
        if set(tree) != {'dirs', 'files'} or '.' not in tree['dirs']:
            raise ValueError('Malformed history inventory')
        if set(tree['dirs']) & set(tree['files']):
            raise ValueError('Duplicated inventory member')
        for name in tree['dirs']:
            if name != '.':
                retention.relative(name)
        for name, rec in tree['files'].items():
            retention.relative(name)
            if len(rec['sha256']) != 64 or rec['size'] < 0:
                raise ValueError('Invalid inventory digest or size')
    absent = plan['absent_roots']
    if len(names) != len(set(names)) or len(absent) != len(set(absent)) or set(names) & set(absent):
        raise ValueError('Duplicated history roots in plan')
    if set(names) | set(absent) != set(HISTORY_ROOTS):
        raise ValueError('Incomplete fixed history scope in plan')
    canonical = project / CANONICAL_RELATIVE
    for name, key in [('cleanup_status.json', 'cleanup_status_sha256'),
                      ('retention_manifest.json', 'retention_manifest_sha256')]:
        if retention.file_sha(canonical / name) != plan[key]:
            raise ValueError('Canonical receipt changed after history plan was made')


def read_events(path, plan):
    if not exists(path):
        return {}
    retention.safe(path)
    known = {x['source']: x for x in plan['roots']}
    state = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        name = event.get('source')
        if (name not in known or event.get('destination') != known[name]['destination']
                or event.get('tree_sha256') != retention.digest(known[name]['tree'])
                or event.get('state') not in ('PENDING', 'MOVED')):
            raise ValueError('Organization journal does not match fixed plan')
        prior = state.get(name)
        if (event['state'] == 'MOVED' and prior != 'PENDING') or prior == 'MOVED':
            raise ValueError('Invalid organization journal transition')
        state[name] = event['state']
    return state


def append_event(path, item, state):
    record = dict(source=item['source'], destination=item['destination'],
                  tree_sha256=retention.digest(item['tree']), state=state,
                  time=retention.now())
    if path.is_symlink():
        raise ValueError('Symlink organization journal')
    with path.open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def check_locations(project, plan, events, *, hash_files=True):
    """Preflight every root before any move; reject collisions and new roots."""
    for name in plan['absent_roots']:
        source, dest = location(project, name)
        if exists(source) or exists(dest):
            raise ValueError('A previously absent root appeared; fixed plan will not expand: ' + name)
    archive = project / ARCHIVE_RELATIVE
    anchor = archive if archive.exists() else archive.parent if archive.parent.exists() else project / 'results'
    archive_device = retention.safe(anchor, directory=True).st_dev
    for item in plan['roots']:
        source = project / item['source']
        dest = project / item['destination']
        at_source, at_dest = exists(source), exists(dest)
        if at_source == at_dest:
            raise ValueError('History is missing or exists at both locations: ' + item['source'])
        current = source if at_source else dest
        if at_dest and events.get(item['source']) not in ('PENDING', 'MOVED'):
            raise ValueError('Unrecorded history movement: ' + item['source'])
        if at_source and events.get(item['source']) == 'MOVED':
            raise ValueError('An archived source was recreated: ' + item['source'])
        if retention.safe(current, directory=True).st_dev != archive_device:
            raise ValueError('Cross-filesystem move is prohibited: ' + item['source'])
        if hash_files:
            check_tree(current, item['tree'])


def rename_function():
    """Resolve no-overwrite rename, with a bounded Linux syscall fallback."""
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, 'renameat2', None)
    if function is not None:
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
    else:
        syscall = getattr(libc, 'syscall', None)
        number = {'x86_64': 316, 'amd64': 316, 'aarch64': 276, 'arm64': 276}.get(platform.machine().lower())
        if syscall is None or number is None or sys.platform != 'linux':
            raise RuntimeError('Linux no-overwrite rename is unavailable; no history was moved')
        syscall.restype = ctypes.c_long

        def function(source_fd, source, dest_fd, dest, flags):
            return syscall(ctypes.c_long(number), ctypes.c_int(source_fd), ctypes.c_char_p(source),
                           ctypes.c_int(dest_fd), ctypes.c_char_p(dest), ctypes.c_uint(flags))
    # Empty path operands cannot rename a file. ENOENT confirms kernel support
    # before the first move rather than discovering ENOSYS halfway through.
    if function(-100, b'', -100, b'', 1) != -1 or ctypes.get_errno() != errno.ENOENT:
        code = ctypes.get_errno()
        raise RuntimeError('No-overwrite rename capability check failed: ' + os.strerror(code))
    return function


def rename_noreplace(source, destination):
    source, destination = Path(source), Path(destination)
    safe_parents(source.parent)
    safe_parents(destination.parent)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with ExitStack() as stack:
        src_fd = os.open(source.parent, flags)
        stack.callback(os.close, src_fd)
        dest_fd = os.open(destination.parent, flags)
        stack.callback(os.close, dest_fd)
        if rename_function()(src_fd, os.fsencode(source.name), dest_fd, os.fsencode(destination.name), 1) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))
        os.fsync(src_fd)
        os.fsync(dest_fd)


def move_one(project, archive, item, journal_path):
    source = project / item['source']
    dest = project / item['destination']
    if not exists(source):
        # Recovery after rename and before its fsynced MOVED receipt.
        check_tree(dest, item['tree'])
        append_event(journal_path, item, 'MOVED')
        return
    if exists(dest):
        raise ValueError('Destination collision; no overwrite: ' + str(dest))
    # Check root identity again immediately before the atomic rename.
    if identity(retention.safe(source, directory=True)) != item['tree']['dirs']['.']:
        raise ValueError('History root identity changed: ' + str(source))
    append_event(journal_path, item, 'PENDING')
    rename_noreplace(source, dest)
    check_tree(dest, item['tree'])
    append_event(journal_path, item, 'MOVED')


def canonical_ready(project):
    canonical = project / CANONICAL_RELATIVE
    retention.safe(canonical, directory=True)
    status = retention.read(canonical / 'cleanup_status.json')
    if status.get('status') != 'CLEANED':
        raise ValueError('Verified cleanup must finish with CLEANED before organizing history')
    report = retention.verify_ready(canonical)
    cleanup = retention.read(canonical / 'cleanup_plan.json')
    count = len(cleanup['files'])
    if status.get('planned_files') != count or status.get('deleted_files') != count:
        raise ValueError('Cleanup count does not match the retained cleanup plan')
    if report.get('project_root') != str(project):
        raise ValueError('Canonical retention belongs to a different project')
    return report


def write_text(path, text):
    if path.is_symlink():
        raise ValueError('Symlink organization index')
    tmp = path.with_name(path.name + '.tmp')
    if exists(tmp):
        raise ValueError('Unexpected temporary organization index: ' + str(tmp))
    with tmp.open('x') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def indexes(project, archive, plan):
    rows = ['old_relative_path\tnew_relative_path\tfile_count\tbytes\ttree_sha256']
    for item in plan['roots']:
        files = item['tree']['files']
        rows.append('\t'.join([item['source'], item['destination'], str(len(files)),
                               str(sum(v['size'] for v in files.values())), retention.digest(item['tree'])]))
    write_text(archive / 'path_mapping.tsv', '\n'.join(rows) + '\n')
    write_text(archive / 'README_CN.md',
        '# 历史实验归档\n\n'
        '此目录集中保存已经结束的历史实验。仅移动目录，未删除、修改或重新训练任何模型。\n\n'
        '当前固定模型、设置和结果位于 `../../que_paper_baselines_20260911/`。'
        '每项历史设置、结果以及剩余文件均保持原字节内容，完整清单见 '
        '`organization_plan.json`，旧路径与新路径对应见 `path_mapping.tsv`。\n\n'
        '**历史流程已退役。请勿重新运行旧训练队列、监测器或旧的保存清理脚本。** '
        '历史记录中的绝对路径保留原样，查阅时按映射定位。'
        '新的整理脚本可用相同固定版本重复执行，以核验或恢复尚未完成的移动。\n\n'
        '`que_comprehensive_reconstruction_20260907` 如存在，仅整目录移动，'
        '本工具没有对它补做权重删除。其他未列入计划的目录保持原位置。\n\n'
        '整理可逆：所有目录内容及 inode 均保留；未来恢复原路径须先确认工作区空闲、'
        '原路径不存在，并按不可变计划逐项核验。不要覆盖任何新生成的同名目录。\n')


def read_status(project):
    project = Path(project).absolute()
    archive = project / ARCHIVE_RELATIVE
    path = archive / STATUS_FILE
    if exists(path):
        retention.safe(path)
        return retention.read(path)
    return dict(status='not_started', archive=str(archive))


def organize(project, execute=False):
    project = Path(project).absolute()
    retention.safe(project, directory=True)
    archive = project / ARCHIVE_RELATIVE
    safe_parents(archive)
    with ExitStack() as stack:
        for name in ('que_history_organization.lock', 'que_paper_closeout.lock', 'que_gpu_7.lock'):
            retention.lock_file(stack, project / 'logs' / name)
        source_focus = project / 'results' / FOCUS_NAME
        dest_focus = archive / FOCUS_NAME
        if exists(source_focus) and exists(dest_focus):
            raise ValueError('Focused experiment exists at both old and archived paths')
        focus = source_focus if exists(source_focus) else dest_focus
        if exists(focus):
            retention.safe(focus, directory=True)
            retention.lock_file(stack, focus / 'campaign.lock')
            status_path = focus / 'campaign_status.json'
            if status_path.exists():
                s = retention.read(status_path)
                if s.get('status') != 'completed_search' or s.get('active_case'):
                    raise RuntimeError('Focused search is not complete')
        retention.assert_idle(project)
        print('核验最终模型、预测、参数及历史备份；通过后才整理目录。', flush=True)
        report = canonical_ready(project)
        plan_path = archive / PLAN_FILE
        if exists(plan_path):
            retention.safe(plan_path)
            plan = retention.read(plan_path)
        else:
            plan = build_plan(project, report)
        validate_plan(plan, project)
        if plan['retention_signature'] != report['signature']:
            raise ValueError('Canonical retention signature changed')
        events = read_events(archive / JOURNAL_FILE, plan)
        check_locations(project, plan, events)
        total_files = sum(len(x['tree']['files']) for x in plan['roots'])
        total_bytes = sum(v['size'] for x in plan['roots'] for v in x['tree']['files'].values())
        result = dict(status='PREVIEW', project_root=str(project), archive=str(archive),
                      planned_directories=len(plan['roots']), file_count=total_files,
                      total_bytes=total_bytes, deleted_files=0,
                      canonical_baselines=str(project / CANONICAL_RELATIVE),
                      plan_signature=plan['signature'], updated_utc=retention.now())
        safe_mkdir(archive)
        if not exists(plan_path):
            # Exclusive creation: never replace the immutable plan.
            with plan_path.open('x') as f:
                f.write(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
                f.flush()
                os.fsync(f.fileno())
        if not execute:
            if not exists(archive / STATUS_FILE):
                write_text(archive / STATUS_FILE, json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            return {**result, 'moves': [{k: x[k] for k in ('source', 'destination')} for x in plan['roots']]}
        rename_function()
        indexes(project, archive, plan)
        retention.assert_idle(project)
        for index, item in enumerate(plan['roots'], 1):
            if events.get(item['source']) != 'MOVED':
                move_one(project, archive, item, archive / JOURNAL_FILE)
            print(f'归档历史目录：{index}/{len(plan["roots"])} {Path(item["source"]).name}', flush=True)
        final_events = read_events(archive / JOURNAL_FILE, plan)
        check_locations(project, plan, final_events, hash_files=False)
        if any(final_events.get(x['source']) != 'MOVED' for x in plan['roots']):
            raise ValueError('Incomplete organization journal')
        result.update(status='ARCHIVED', moved_directories=len(plan['roots']),
                      updated_utc=retention.now(), history_absolute_paths_rewritten=False,
                      retired_workflows_must_not_be_restarted=True)
        write_text(archive / STATUS_FILE, json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.home() / 'projects/STaR-GNN-BWDF')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = read_status(args.project_root) if args.status else organize(args.project_root, args.execute)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr, flush=True)
        print('未删除任何文件；已完成移动的路径记录在归档日志中，可使用相同工具恢复。', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
