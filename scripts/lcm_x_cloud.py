#!/usr/bin/env python3
"""Stage or publish a read-only capture snapshot using the directory transport."""
import argparse
import json
from pathlib import Path
import sys
import types


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('snapshot', 'publish'):
        sub = commands.add_parser(command)
        sub.add_argument('--home', required=True, type=Path)
        sub.add_argument('--copy-id', required=True)
        sub.add_argument('--bucket-dir', required=True, type=Path)
        sub.add_argument('--prefix', required=True)
        sub.add_argument('--quiet-s', type=float, default=0)
        sub.add_argument('--part-size', type=int, default=33554432)
        if command == 'publish':
            sub.add_argument('--timer', action='store_true')
    args = parser.parse_args(argv)
    if 'hermes_lcm' not in sys.modules:
        package = types.ModuleType('hermes_lcm')
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        package.__package__ = 'hermes_lcm'
        sys.modules['hermes_lcm'] = package
    from hermes_lcm import cloud_copy
    try:
        prefix = args.prefix + '/' + args.copy_id
        transport = cloud_copy.DirTransport(args.bucket_dir, prefix)
        manifest = cloud_copy.publish(args.home, args.copy_id, transport, prefix,
            quiet_s=args.quiet_s, part_size=args.part_size, timer=getattr(args, 'timer', False),
            snapshot_only=args.command == 'snapshot')
        result = dict(code='done', copy_id=manifest['copy_id'], generation=manifest['generation'],
                      file_bytes=manifest['database']['file_bytes'], payloads=len(manifest['payloads']),
                      unresolved_minted=len(manifest['unresolved_minted']), unresolved_quoted=len(manifest['unresolved_quoted']))
        if 'blocker' in manifest:
            result['blocker'] = manifest['blocker']
        status = 0
    except Exception as error:
        code = getattr(error, 'code', 'error')
        result = {'code': code}
        detail = getattr(error, 'detail', {})
        if 'generation' in detail:
            result['generation'] = detail['generation']
        status = (3 if code in cloud_copy.SKIPS else 2) if hasattr(error, 'code') else 1
    print(json.dumps(result, sort_keys=True))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
