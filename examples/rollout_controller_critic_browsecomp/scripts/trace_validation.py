"""Dedicated-GPU validation of immutable warmup exports."""
from critic_replica import FrozenCriticReplica
from critic_equivalence import compare_scores
from trace_warmup import report


class TraceValidationReplica(FrozenCriticReplica):
    def evaluate(self, directory, version, rows, indices, native, constant, root_mass, tolerance):
        publication = self.publish(directory, version)
        self.begin(version)
        try:
            full = self.score([r['context'] for r in rows], version)
            probe = dict(version=version, scores=[full['scores'][i] for i in indices])
            repeat = self.score([rows[i]['context'] for i in indices], version)
            comparison = compare_scores(native, probe, repeat, version=version,
                                        count=len(indices), tolerance=tolerance)
            if not comparison['passed']:
                return dict(publication=publication, comparison=comparison, rejected=True)
            return dict(publication=publication, comparison=comparison,
                        predictions=full['scores'],
                        report=report(rows, full['scores'], constant, root_mass))
        finally:
            self.end(version)
