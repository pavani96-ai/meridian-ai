"""Document storage tests using temporary files and mocked cloud storage."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from exceptions import ConfigurationError
from rag import document_store


@pytest.fixture
def storage_settings(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        is_cloud=False, local_upload_dir=str(tmp_path / "uploads"),
        GCS_BUCKET_NAME="test-bucket", GCS_PREFIX="uploads/", GCP_PROJECT="",
    )
    monkeypatch.setattr(document_store, "settings", settings)
    return settings


def test_local_save_copies_contents_and_strips_parent_path(storage_settings, tmp_path):
    original = tmp_path / "original.pdf"
    original.write_bytes(b"%PDF-1.4 test")
    stored = document_store.save(str(original), "../invoice.pdf")
    assert stored == str(Path(storage_settings.local_upload_dir) / "invoice.pdf")
    assert Path(stored).read_bytes() == original.read_bytes()
    assert document_store.location() == storage_settings.local_upload_dir


def test_local_pdfs_filters_and_sorts(storage_settings):
    folder = Path(storage_settings.local_upload_dir)
    folder.mkdir()
    for name in ["b.PDF", "a.pdf", "notes.txt"]:
        (folder / name).write_bytes(b"test")
    with document_store.pdfs() as files:
        assert list(files) == [
            (str(folder / "a.pdf"), "local://a.pdf"),
            (str(folder / "b.PDF"), "local://b.PDF"),
        ]


def test_missing_local_folder_is_created(storage_settings):
    with document_store.pdfs() as files:
        assert list(files) == []
    assert Path(storage_settings.local_upload_dir).is_dir()


def test_cloud_save_uses_bucket_and_pdf_content_type(storage_settings, monkeypatch, tmp_path):
    storage_settings.is_cloud = True
    bucket = Mock()
    monkeypatch.setattr(document_store, "_bucket", lambda: bucket)
    path = str(tmp_path / "file.pdf")
    assert document_store.save(path, "../file.pdf") == "gs://test-bucket/uploads/file.pdf"
    bucket.blob.assert_called_once_with("uploads/file.pdf")
    bucket.blob.return_value.upload_from_filename.assert_called_once_with(path, content_type="application/pdf")
    assert document_store.location() == "gs://test-bucket/uploads/"


def test_cloud_downloads_pdfs_and_cleans_up_on_error(storage_settings, monkeypatch):
    storage_settings.is_cloud = True
    blob = Mock()
    blob.name = "uploads/invoice.pdf"
    blob.download_to_filename.side_effect = lambda path: Path(path).write_bytes(b"pdf")
    other = Mock()
    other.name = "uploads/notes.txt"
    bucket = Mock()
    bucket.list_blobs.return_value = [blob, other]
    monkeypatch.setattr(document_store, "_bucket", lambda: bucket)
    with pytest.raises(RuntimeError, match="consumer failed"):
        with document_store.pdfs() as files:
            [(path, source)] = list(files)
            assert Path(path).read_bytes() == b"pdf"
            assert source == "gs://test-bucket/uploads/invoice.pdf"
            raise RuntimeError("consumer failed")
    assert not Path(path).parent.exists()
    other.download_to_filename.assert_not_called()
    bucket.list_blobs.assert_called_once_with(prefix="uploads/")


def test_cloud_requires_bucket_name(storage_settings):
    storage_settings.is_cloud = True
    storage_settings.GCS_BUCKET_NAME = ""
    with pytest.raises(ConfigurationError, match="GCS_BUCKET_NAME"):
        document_store._bucket()
