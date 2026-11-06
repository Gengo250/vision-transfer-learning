import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "sync.py"
spec = importlib.util.spec_from_file_location("sync_under_test", SCRIPT)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def command(directory, *args):
    result = subprocess.run(["git", *args], cwd=directory, text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


class SynchronizationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        command(self.base, "init", "--bare", "--initial-branch=main", "remote.git")
        command(self.base, "clone", str(self.base / "remote.git"), "one")
        self.one = self.base / "one"
        command(self.one, "config", "user.name", "Test")
        command(self.one, "config", "user.email", "test@example.com")
        (self.one / ".gitignore").write_text("ignored/\n")
        (self.one / "file.txt").write_text("initial")
        command(self.one, "add", ".")
        command(self.one, "commit", "-m", "initial")
        command(self.one, "push", "origin", "main")
        command(self.base, "clone", str(self.base / "remote.git"), "two")
        self.two = self.base / "two"
        command(self.two, "config", "user.name", "Test")
        command(self.two, "config", "user.email", "test@example.com")
        sync.REPO = self.one
        sync.LOCK = self.base / ".one.sync-lock"
        self.env = os.environ.copy()
        self.env["GIT_TERMINAL_PROMPT"] = "0"

    def tearDown(self):
        self.tmp.cleanup()

    def remote_change(self):
        (self.two / "remote.txt").write_text("remote")
        command(self.two, "add", ".")
        command(self.two, "commit", "-m", "remote change")
        command(self.two, "push", "origin", "main")

    def test_push_and_ignore(self):
        (self.one / "file.txt").write_text("local")
        (self.one / "ignored").mkdir()
        (self.one / "ignored" / "secret").write_text("never push")
        sync.sync_once(self.env, False)
        command(self.two, "pull", "--ff-only")
        self.assertEqual((self.two / "file.txt").read_text(), "local")
        self.assertFalse((self.two / "ignored").exists())
        self.assertEqual(command(self.one, "status", "--porcelain"), "")
        before = command(self.one, "rev-parse", "HEAD")
        sync.sync_once(self.env, False)
        self.assertEqual(before, command(self.one, "rev-parse", "HEAD"))

    def test_fast_forward_only_at_start(self):
        self.remote_change()
        sync.initialize(self.env, False)
        self.assertEqual((self.one / "remote.txt").read_text(), "remote")

    def test_remote_and_dirty_stops_without_modifying_local(self):
        (self.two / "file.txt").write_text("remote work")
        command(self.two, "add", "file.txt")
        command(self.two, "commit", "-m", "remote change")
        command(self.two, "push", "origin", "main")
        (self.one / "file.txt").write_text("unsent work")
        with self.assertRaises(sync.SyncError):
            sync.initialize(self.env, False)
        self.assertEqual((self.one / "file.txt").read_text(), "unsent work")
        self.assertFalse((self.one / "remote.txt").exists())

    def test_fast_forward_preserves_dirty_file_not_changed_remotely(self):
        self.remote_change()
        (self.one / "file.txt").write_text("unsent work")
        sync.initialize(self.env, False)
        self.assertEqual((self.one / "file.txt").read_text(), "unsent work")
        self.assertEqual((self.one / "remote.txt").read_text(), "remote")

    def test_output_only_notebook_changes_are_not_committed_or_deleted(self):
        notebook = self.one / "notebook.ipynb"
        base = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
            {"cell_type": "code", "source": ["print(1)"], "metadata": {},
             "execution_count": None, "outputs": []}]}
        notebook.write_text(json.dumps(base))
        command(self.one, "add", "notebook.ipynb")
        command(self.one, "commit", "-m", "notebook")
        command(self.one, "push", "origin", "main")
        base["cells"][0]["execution_count"] = 1
        base["cells"][0]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["1\\n"]}]
        notebook.write_text(json.dumps(base))
        (self.one / "file.txt").write_text("code in another file")
        before = notebook.read_bytes()
        sync.sync_once(self.env, False)
        self.assertEqual(notebook.read_bytes(), before)
        self.assertIn("notebook.ipynb", command(self.one, "status", "--porcelain"))
        self.assertEqual(command(self.one, "diff", "--name-only", "HEAD~1", "HEAD"), "file.txt")
        head = command(self.one, "rev-parse", "HEAD")
        sync.sync_once(self.env, False)
        self.assertEqual(command(self.one, "rev-parse", "HEAD"), head)

    def test_notebook_code_changes_and_outputs_are_published_together(self):
        notebook = self.one / "notebook.ipynb"
        nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
            {"cell_type": "code", "source": ["print(1)"], "metadata": {},
             "execution_count": None, "outputs": []}]}
        notebook.write_text(json.dumps(nb))
        command(self.one, "add", "notebook.ipynb")
        command(self.one, "commit", "-m", "notebook")
        command(self.one, "push", "origin", "main")
        nb["cells"][0]["source"] = ["print(2)"]
        nb["cells"][0]["execution_count"] = 2
        nb["cells"][0]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["2\\n"]}]
        notebook.write_text(json.dumps(nb))
        sync.sync_once(self.env, False)
        self.assertEqual(json.loads(command(self.one, "show", "HEAD:notebook.ipynb")), nb)

    def test_execution_metadata_is_ignored_but_other_metadata_is_preserved(self):
        raw = {"nbformat": 4, "metadata": {}, "cells": [
            {"cell_type": "code", "source": ["x = 1"], "metadata": {}, "outputs": [], "execution_count": None}]}
        changed = json.loads(json.dumps(raw))
        changed["cells"][0]["metadata"]["execution"] = {"iopub.execute_input": "now"}
        self.assertEqual(sync.notebook_without_volatile_outputs(json.dumps(raw).encode()),
                         sync.notebook_without_volatile_outputs(json.dumps(changed).encode()))
        changed["cells"][0]["metadata"]["tags"] = ["important"]
        self.assertNotEqual(sync.notebook_without_volatile_outputs(json.dumps(raw).encode()),
                            sync.notebook_without_volatile_outputs(json.dumps(changed).encode()))

    def test_recovery_cell_fast_forwards_without_losing_dirty_notebook(self):
        (self.one / "transfer_learning.ipynb").write_text('{"cells": [], "metadata": {}}')
        command(self.one, "add", "transfer_learning.ipynb")
        command(self.one, "commit", "-m", "add notebook")
        command(self.one, "push", "origin", "main")
        command(self.two, "pull", "--ff-only")
        (self.two / "sync.py").write_text("# remote helper")
        (self.two / "README.md").write_text("remote instructions")
        command(self.two, "add", "sync.py", "README.md")
        command(self.two, "commit", "-m", "new synchronizer")
        command(self.two, "push", "origin", "main")
        modified = '{"cells": [], "metadata": {"unsent": true}}'
        (self.one / "transfer_learning.ipynb").write_text(modified)
        doc = (SCRIPT.parent / "COLAB_RECUPERACAO.md").read_text()
        recovery = doc.split('```python')[2].split('```')[0]
        recovery = recovery.replace("Path('/content/drive/MyDrive/vision-transfer-learning')", f"Path({str(self.one)!r})")
        recovery = recovery.replace('https://github.com/Gengo250/vision-transfer-learning.git', str(self.base / 'remote.git'))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(recovery, 'recovery-cell', 'exec'), {})
        self.assertEqual((self.one / 'transfer_learning.ipynb').read_text(), modified)
        self.assertEqual((self.one / 'sync.py').read_text(), '# remote helper')
        self.assertEqual((self.one / 'README.md').read_text(), 'remote instructions')
        self.assertEqual(command(self.one, 'rev-list', '--left-right', '--count', 'HEAD...origin/main'), '0\t0')

    def test_recovery_cell_refuses_overlapping_modification(self):
        self.remote_change()
        (self.one / 'remote.txt').write_text('do not overwrite')
        doc = (SCRIPT.parent / 'COLAB_RECUPERACAO.md').read_text()
        recovery = doc.split('```python')[2].split('```')[0]
        recovery = recovery.replace("Path('/content/drive/MyDrive/vision-transfer-learning')", f"Path({str(self.one)!r})")
        recovery = recovery.replace('https://github.com/Gengo250/vision-transfer-learning.git', str(self.base / 'remote.git'))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            exec(compile(recovery, 'recovery-cell', 'exec'), {})
        self.assertEqual((self.one / 'remote.txt').read_text(), 'do not overwrite')

    def test_remote_change_stops_active_monitor_even_when_clean(self):
        self.remote_change()
        with self.assertRaises(sync.SyncError):
            sync.sync_once(self.env, False)
        self.assertFalse((self.one / "remote.txt").exists())

    def test_staged_changes_remain_untouched(self):
        (self.one / "file.txt").write_text("staged")
        command(self.one, "add", "file.txt")
        before = command(self.one, "diff", "--cached")
        with self.assertRaises(sync.SyncError):
            sync.sync_once(self.env, False)
        self.assertEqual(command(self.one, "diff", "--cached"), before)

    def test_lock_refuses_concurrency(self):
        with sync.repository_lock():
            with self.assertRaises(sync.SyncError):
                with sync.repository_lock():
                    pass
        self.assertFalse(sync.LOCK.exists())

    def test_ignored_tracked_modification_is_blocked(self):
        (self.one / ".gitignore").write_text("ignored/\nfile.txt\n")
        (self.one / "file.txt").write_text("changed")
        with self.assertRaises(sync.SyncError):
            sync.sync_once(self.env, False)
        self.assertEqual((self.one / "file.txt").read_text(), "changed")

    def test_merge_never_overwrites_ignored_file(self):
        (self.one / "ignored").mkdir()
        (self.one / "ignored" / "saved").write_text("local important data")
        (self.two / "ignored").mkdir()
        (self.two / "ignored" / "saved").write_text("remote data")
        command(self.two, "add", "-f", "ignored/saved")
        command(self.two, "commit", "-m", "add ignored file")
        command(self.two, "push", "origin", "main")
        with self.assertRaises(sync.SyncError):
            sync.initialize(self.env, False)
        self.assertEqual((self.one / "ignored" / "saved").read_text(), "local important data")

    def test_ignored_file_in_unpushed_commit_blocks_push(self):
        (self.one / "ignored").mkdir()
        (self.one / "ignored" / "secret").write_text("sensitive")
        command(self.one, "add", "-f", "ignored/secret")
        command(self.one, "commit", "-m", "manual commit")
        with self.assertRaises(sync.SyncError):
            sync.sync_once(self.env, False)
        self.assertEqual(command(self.one, "rev-list", "--count", "origin/main..HEAD"), "1")

    def test_ignored_file_added_and_removed_in_local_history_blocks_push(self):
        (self.one / "ignored").mkdir()
        (self.one / "ignored" / "secret").write_text("sensitive")
        command(self.one, "add", "-f", "ignored/secret")
        command(self.one, "commit", "-m", "add ignored")
        command(self.one, "rm", "ignored/secret")
        command(self.one, "commit", "-m", "remove ignored")
        with self.assertRaises(sync.SyncError):
            sync.sync_once(self.env, False)
        self.assertEqual(command(self.one, "rev-list", "--count", "origin/main..HEAD"), "2")

    def test_concurrent_push_rejection_preserves_local_commit(self):
        (self.one / "file.txt").write_text("local pending")
        original_git = sync.git
        def git_with_competing_push(*args, **kwargs):
            if "push" in args:
                self.remote_change()
            return original_git(*args, **kwargs)
        with mock.patch.object(sync, "git", side_effect=git_with_competing_push):
            with self.assertRaises(sync.SyncError):
                sync.sync_once(self.env, False)
        self.assertEqual((self.one / "file.txt").read_text(), "local pending")
        self.assertEqual(command(self.one, "rev-list", "--count", "origin/main..HEAD"), "1")
        self.assertEqual(command(self.two, "show", "HEAD:remote.txt"), "remote")

    def test_colab_rejects_different_push_url_before_network(self):
        command(self.one, "remote", "set-url", "origin", "https://github.com/Gengo250/vision-transfer-learning.git")
        command(self.one, "remote", "set-url", "--push", "origin", "https://example.com/other.git")
        with sync.authentication("test-token") as env:
            with self.assertRaises(sync.SyncError):
                sync.check_repo(env, True)

    def test_colab_token_helper_is_temporary_and_never_in_remote(self):
        token = "test-secret-not-a-real-token"
        with sync.authentication(token) as env:
            helper = Path(env["GIT_ASKPASS"])
            self.assertTrue(helper.exists())
            self.assertNotIn(token, helper.read_text())
            self.assertNotIn(token, command(self.one, "remote", "get-url", "origin"))
            if os.name != "nt":
                value = subprocess.check_output([str(helper), "Password for github"], env=env, text=True).strip()
                self.assertEqual(value, token)
        self.assertFalse(helper.exists())

    def test_content_fingerprint_changes_for_already_modified_file(self):
        (self.one / "file.txt").write_text("first")
        first = sync.fingerprint(self.env)
        (self.one / "file.txt").write_text("second")
        self.assertNotEqual(first, sync.fingerprint(self.env))


if __name__ == "__main__":
    unittest.main()
