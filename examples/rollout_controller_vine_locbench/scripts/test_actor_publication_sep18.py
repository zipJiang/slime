"""Regression: native resume must not publish the last-loaded reference model."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def load_actor(monkeypatch):
    class Parent:
        def _switch_model(self, tag):
            self.switches.append(tag)
            self.parameter = self.backups[tag]
            self._active_model_tag = tag

        def update_weights(self):
            # Model the distributed transport: publish LIVE weights, not backups.
            if self.args.offload_train:
                self._switch_model('actor')
            self.published = self.parameter
            self.weight_updater.weight_version += 1
            return 'published'

    module = ModuleType('vine_actor')
    module.VineActor = Parent
    monkeypatch.setitem(sys.modules, 'vine_actor', module)
    torch = ModuleType('torch')
    distributed = ModuleType('torch.distributed')
    distributed.get_rank = lambda: 0
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.distributed', distributed)
    spec = importlib.util.spec_from_file_location('publication_under_test', Path(__file__).with_name('vine_actor_publication.py'))
    publication = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publication)
    return publication.VineActor


def instance(actor, tmp_path, active, offload=False):
    obj = actor()
    obj.args = SimpleNamespace(offload_train=offload, save=str(tmp_path/'actor'))
    obj.backups = {'actor': 7.0, 'ref': 2.0}
    obj.parameter = obj.backups[active]
    obj._active_model_tag = active
    obj.switches = []
    obj.weight_updater = SimpleNamespace(weight_version=0)
    return obj


def test_resumed_reference_is_replaced_before_publication(monkeypatch, tmp_path):
    obj = instance(load_actor(monkeypatch), tmp_path, 'ref')
    assert obj.update_weights() == 'published'
    assert obj.published == 7.0
    assert obj.switches == ['actor']
    assert (tmp_path/'actor-publication-audit/version-1-rank-0.json').is_file()


def test_current_actor_is_not_copied_again(monkeypatch, tmp_path):
    obj = instance(load_actor(monkeypatch), tmp_path, 'actor')
    obj.update_weights()
    assert obj.published == 7.0
    assert obj.switches == []


def test_offloaded_parent_retains_its_own_wakeup(monkeypatch, tmp_path):
    obj = instance(load_actor(monkeypatch), tmp_path, 'ref', offload=True)
    obj.update_weights()
    assert obj.published == 7.0
    assert obj.switches == ['actor']
