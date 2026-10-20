"""Materialize a portable report and preview it over an SSH-forwardable port."""

import argparse
import functools
import hashlib
import html
from html.parser import HTMLParser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {'src', 'href'} and value and not urlsplit(value).scheme and not value.startswith('#'):
                self.references.append(value)


def package(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new preview directory')
    document = source.read_text()
    parser = References()
    parser.feed(document)
    files, substitutions = {}, {}
    for ref in dict.fromkeys(parser.references):
        parsed = urlsplit(ref)
        path = (source.parent/unquote(parsed.path)).resolve()
        if not path.is_file() or not path.is_relative_to(ROOT):
            raise ValueError(f'Report reference is not an existing workspace file: {ref}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        name = f'assets/{digest[:16]}-{path.name}'
        substitutions[ref] = name + (f'?{parsed.query}' if parsed.query else '') + (f'#{parsed.fragment}' if parsed.fragment else '')
        files[name] = {'source': str(path.relative_to(ROOT)), 'sha256': digest, 'bytes': path.stat().st_size}
    (output/'assets').mkdir(parents=True)
    for name, record in files.items():
        shutil.copyfile(ROOT/record['source'], output/name)
    for before, after in substitutions.items():
        for quote in ['"', "'"]:
            document = document.replace(quote+html.escape(before, quote=True)+quote, quote+after+quote)
    (output/'index.html').write_text(document)
    parser = References()
    parser.feed(document)
    if any(not (output/unquote(urlsplit(ref).path)).is_file() for ref in parser.references):
        raise ValueError('Portable report contains an unresolved local reference')
    (output/'manifest.json').write_text(json.dumps({'original_report': str(source.relative_to(ROOT)),
        'original_report_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'assets': files, 'local_references': len(parser.references), 'all_assets_materialized': True}, indent=2)+'\n')
    (output/'README.txt').write_text('Open index.html with Safari or Chrome to view this report offline.\n'
        'Keep the assets folder beside index.html; it contains the videos and figures.\n')
    paths = [path for path in output.rglob('*') if path.is_file()]
    archive = output/'curobo-report.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zipped:
        for path in paths:
            zipped.write(path, Path('curobo-report')/path.relative_to(output))
    print(json.dumps({'preview_directory': str(output), 'assets': len(files),
                      'zip_bytes': archive.stat().st_size}), flush=True)
    return output


class PreviewHandler(SimpleHTTPRequestHandler):
    """Support byte ranges so Safari can seek through the saved MP4 files."""

    def send_head(self):
        self.byte_range = None
        requested_range = self.headers.get('Range')
        path = Path(self.translate_path(self.path))
        if not requested_range or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested_range)
        size = path.stat().st_size
        if match is None or not any(match.groups()) or not size:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None
        a, b = match.groups()
        start = int(a) if a else max(0, size-int(b))
        end = min(int(b), size-1) if a and b else size-1
        if not 0 <= start <= end < size:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None
        stream = path.open('rb')
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end-start+1))
        self.end_headers()
        self.byte_range = (start, end)
        return stream

    def copyfile(self, source, outputfile):
        if self.byte_range is None:
            return super().copyfile(source, outputfile)
        start, end = self.byte_range
        source.seek(start)
        remaining = end-start+1
        while remaining:
            data = source.read(min(remaining, 64*1024))
            if not data:
                break
            outputfile.write(data)
            remaining -= len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='results/task-grounding-curobo-report/index.html')
    parser.add_argument('--output', default='results/curobo-report-preview')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--package-only', action='store_true')
    args = parser.parse_args()
    output = package(ROOT/args.source, ROOT/args.output)
    if not args.package_only:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), functools.partial(PreviewHandler, directory=str(output)))
        print(f'Report preview ready at http://127.0.0.1:{args.port}/', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
