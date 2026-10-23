"""Local server for the site, with HTTP Range support so video can seek.

Python's built-in `http.server` ignores Range headers; browsers then cannot
seek inside the 4K clips. This adds byte-range responses and nothing else.

    python serve.py            # http://localhost:5178
    python serve.py 8080       # another port
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      '.mp4': 'video/mp4', '.woff2': 'font/woff2', '.js': 'text/javascript'}

    def _etag(self, path):
        st = os.stat(path)
        return f'"{st.st_size:x}-{int(st.st_mtime):x}"'

    def send_head(self):
        rng = self.headers.get('Range')
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        # If-Range: a browser holding pieces of an older version of the file
        # (e.g. after a re-export) gets the whole new file, not a bad range.
        if_range = self.headers.get('If-Range')
        if if_range and if_range.strip() != self._etag(path) \
                and if_range.strip() != self.date_time_string(int(os.stat(path).st_mtime)):
            return super().send_head()
        m = re.match(r'bytes=(\d*)-(\d*)$', rng.strip())
        size = os.path.getsize(path)
        if not m:
            self.send_error(416)
            return None
        a, b = m.groups()
        if a == '':                       # suffix range: last N bytes
            start, end = max(0, size - int(b)), size - 1
        else:
            start, end = int(a), (int(b) if b else size - 1)
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.end_headers()
            return None
        f = open(path, 'rb')
        f.seek(start)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('ETag', self._etag(path))
        self.send_header('Last-Modified', self.date_time_string(int(os.stat(path).st_mtime)))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, '_remaining', None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        try:
            while remaining > 0:
                chunk = source.read(min(1 << 20, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass    # the browser cancelled a range it no longer needs
        finally:
            self._remaining = None

    def end_headers(self):
        if self.command in ('GET', 'HEAD') and 'Range' not in self.headers:
            self.send_header('Accept-Ranges', 'bytes')
            path = self.translate_path(self.path)
            if os.path.isfile(path):
                self.send_header('ETag', self._etag(path))
        # local project: always revalidate, so re-exported media is picked up
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5178
    print(f'Serving on http://localhost:{port}  (Ctrl+C to stop)')
    ThreadingHTTPServer(('127.0.0.1', port), RangeHandler).serve_forever()
