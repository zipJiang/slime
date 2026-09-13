"""HTTP bridge from the Python 3.13 collector to Slime's critic Ray actors."""
import copy
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from batches import partition_data, put_packets, training_data
from targets import checkpoint_fields


class CriticScorer:
    """Serializes TP collectives and verifies version and TP replica agreement."""
    def __init__(self, actors, args, parallel_config, tokenizer, sentinel_token_id):
        self.actors = actors
        self.args = args
        self.parallel_config = parallel_config
        self.tokenizer = tokenizer
        self.sentinel_token_id = sentinel_token_id
        self.version = None
        self.lock = threading.Lock()

    def begin(self, version):
        import ray
        with self.lock:
            if self.version is not None:
                raise RuntimeError('A critic scoring session is already active')
            versions = ray.get([a.begin_scoring.remote(version) for a in self.actors])
            if versions != [version] * len(self.actors):
                raise RuntimeError('Critic ranks disagree on scoring version')
            self.version = version

    def end(self):
        import ray
        with self.lock:
            if self.version is None:
                raise RuntimeError('No critic scoring session')
            ray.get([a.end_scoring.remote(self.version) for a in self.actors])
            self.version = None

    def score(self, contexts, version):
        import ray
        if not contexts or any(not isinstance(c, str) or not c for c in contexts):
            raise ValueError('Expected nonempty checkpoint contexts')
        with self.lock:
            if self.version is None or version != self.version:
                raise ValueError('Stale or inactive critic version')
            count = len(contexts)
            # Inference only: replicate inputs when a small request cannot fill
            # all DP ranks. Remove duplicates from the response; never train them.
            contexts = list(contexts)
            while len(contexts) < self.parallel_config['dp_size']:
                contexts.append(contexts[-1])
            rows = []
            for i, context in enumerate(contexts):
                record = dict(context=context, target=0., group_index=i,
                              metadata=dict(lane='critic', node_id=i, value_version=version))
                rows.append(checkpoint_fields(record, self.tokenizer,
                            sentinel_token_id=self.sentinel_token_id,
                            max_sequence_length=self.args.seq_length))
            local_args = copy.copy(self.args)
            local_args.global_batch_size = len(rows)
            data = training_data(rows, lane='critic', expected_groups=range(len(rows)))
            refs = put_packets(partition_data(local_args, self.parallel_config, data))
            outputs = ray.get([a.score_checkpoints.remote(refs, version) for a in self.actors])
            values = {}
            for output in outputs:
                if output['version'] != version:
                    raise RuntimeError('Critic result version mismatch')
                for index, value in zip(output['partition'], output['values'], strict=True):
                    if not math.isfinite(value):
                        raise ValueError('Nonfinite critic prediction')
                    if index in values and not math.isclose(values[index], value, abs_tol=1e-5, rel_tol=1e-5):
                        raise RuntimeError('TP replicas disagree on checkpoint prediction')
                    values[index] = value
            if set(values) != set(range(len(rows))):
                raise RuntimeError('Missing checkpoint predictions')
            return dict(version=version, scores=[values[i] for i in range(count)])


def serve(scorer, host='0.0.0.0', port=0):
    """Start a local-cluster service; caller owns shutdown and scorer lifecycle."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                if self.path != '/score':
                    raise ValueError('Unknown endpoint')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError('Invalid scoring request size')
                request = json.loads(self.rfile.read(length))
                result = scorer.score(request['contexts'], request['version'])
                status = 200
            except Exception as exc:
                result, status = dict(error=f'{type(exc).__name__}: {exc}'), 500
            payload = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
