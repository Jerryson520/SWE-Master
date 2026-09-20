"""镜像清理范围回归测试：全部使用 mock，不访问 Docker。"""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call

path = Path(__file__).resolve().parents[1] / 'scripts/collect_teacher_rollouts.py'
spec = importlib.util.spec_from_file_location('cleanup_collector', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CleanupTests(unittest.TestCase):
    def test_only_batch_tags_with_safety_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Mock()
            client.containers.list.return_value = [SimpleNamespace(attrs={'Image': 'busy-id'})]
            manifest = {'initial_images': [{'id': 'old-id', 'tags': ['old:tag']}]}
            images = module.Images(client, Path(directory), Path(directory), manifest, 10)
            images.owned = {'new:tag': 'new-id', 'changed:tag': 'previous-id'}
            objects = {
                'old:tag': SimpleNamespace(id='old-id', tags=['old:tag']),
                'new:tag': SimpleNamespace(id='new-id', tags=['new:tag']),
                'busy:tag': SimpleNamespace(id='busy-id', tags=['busy:tag']),
                'shared:tag': SimpleNamespace(id='shared-id', tags=['shared:tag', 'other:tag']),
                'changed:tag': SimpleNamespace(id='different-id', tags=['changed:tag']),
                'missing:tag': None,
                'unrelated:tag': SimpleNamespace(id='other-id', tags=['unrelated:tag']),
            }
            images.get = Mock(side_effect=objects.get)
            tasks = [{'docker_image': t} for t in objects if t != 'unrelated:tag']
            tasks.append({'docker_image': 'old:tag'})
            images.cleanup(tasks)
            self.assertEqual(client.images.remove.call_args_list, [
                call('new:tag', force=False, noprune=True),
                call('old:tag', force=False, noprune=True)])
            self.assertNotIn(call('unrelated:tag'), images.get.call_args_list)

    def test_failed_delete_is_not_forced(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Mock()
            client.containers.list.return_value = []
            client.images.remove.side_effect = RuntimeError('busy')
            images = module.Images(client, Path(directory), Path(directory), {'initial_images': []}, 10)
            images.get = Mock(return_value=SimpleNamespace(id='id', tags=['task:tag']))
            images.cleanup([{'docker_image': 'task:tag'}])
            client.images.remove.assert_called_once_with('task:tag', force=False, noprune=True)


if __name__ == '__main__':
    unittest.main()
