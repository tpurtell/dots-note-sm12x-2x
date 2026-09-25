#!/usr/bin/env python3
"""Build a native release wrapper around the exact warmed parent image."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle', required=True)
p.add_argument('--image', required=True, help='Qualified native runtime image ID or tag')
p.add_argument('--tag', required=True, help='Versioned GHCR target tag; does not publish')
p.add_argument('--development-without-reasoning-parser', action='store_true',
    help='Build an unqualified development wrapper around a legacy parser-off parent')
a = p.parse_args()
bundle = Path(a.bundle).resolve()
manifest = json.loads((bundle / 'manifest.json').read_text())
image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', a.image]))[0]
if image['Id'] != manifest['source_image_id'] or image['Architecture'] != manifest['architecture']:
    raise SystemExit('Parent image must exactly match the warmed image and native architecture')
parent_tag = 'dots3-release-parent:' + image['Id'].split(':', 1)[1]
subprocess.run(['docker', 'tag', image['Id'], parent_tag], check=True)
root = Path(__file__).resolve().parent
# Temporary build context lives next to the project cache bundle, never /mnt/scratch.
with tempfile.TemporaryDirectory(prefix='.release-build-', dir=bundle.parent) as tmp:
    ctx = Path(tmp)
    for name in ['Dockerfile', 'entrypoint.sh', 'cache_bundle.py', 'verify_reasoning.py']:
        shutil.copy2(root / name, ctx / name)
    shutil.copytree(bundle, ctx / 'cache-seed')
    subprocess.run(['docker', 'build', '--platform', f"linux/{manifest['architecture']}",
        '--build-arg', f'RUNTIME_IMAGE={parent_tag}',
        '--build-arg', 'REASONING_PARSER=' + ('none' if a.development_without_reasoning_parser else 'dots3'),
        '--build-arg', 'REASONING_PARSER_SHA256=' + hashlib.sha256((root.parent / 'dots3_reasoning_parser.py').read_bytes()).hexdigest(),
        '--build-arg', f'RECIPE_REVISION={manifest["recipe_revision"]}',
        '--build-arg', f'PLATFORM={manifest["platform"]}', '-t', a.tag, str(ctx)], check=True)
print(json.dumps({'tag': a.tag, 'parent': image['Id'], 'published': False, 'reasoning_parser': 'none' if a.development_without_reasoning_parser else 'dots3'}))
