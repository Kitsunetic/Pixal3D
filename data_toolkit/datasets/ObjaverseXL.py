import argparse
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from pathlib import Path, PurePosixPath

import pandas as pd

try:
    import objaverse.xl as oxl
except ModuleNotFoundError as error:
    if error.name not in {"objaverse", "objaverse.xl"}:
        raise

    class _MissingObjaverseXL:
        @staticmethod
        def get_annotations():
            raise ModuleNotFoundError(
                "ObjaverseXL requires the optional 'objaverse' package"
            )

        @staticmethod
        def download_objects(*args, **kwargs):
            raise ModuleNotFoundError(
                "ObjaverseXL requires the optional 'objaverse' package"
            )

    oxl = _MissingObjaverseXL()


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _expected_digest(row) -> str:
    content_digest = row.get("content_sha256")
    value = (
        content_digest
        if isinstance(content_digest, str) and content_digest
        else row["sha256"]
    )
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"invalid Objaverse SHA-256: {value!r}")
    return value


def _canonical_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError(f"unsafe Objaverse path: {value!r}")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or ".." in relative.parts
        or "." in relative.parts
    ):
        raise ValueError(f"unsafe Objaverse path: {value!r}")
    return relative


@contextmanager
def _open_regular_file(root: Path, relative: PurePosixPath) -> Iterator[int]:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    try:
        current = os.open(root, directory_flags)
        descriptors.append(current)
        for component in relative.parts[:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            descriptors.append(current)
        asset = os.open(relative.parts[-1], file_flags, dir_fd=current)
        descriptors.append(asset)
        if not stat.S_ISREG(os.fstat(asset).st_mode):
            raise ValueError(f"Objaverse path is not a regular file: {relative}")
        yield asset
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError(f"unsafe Objaverse file: {relative}") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _split_archive_path(
    relative: PurePosixPath,
) -> tuple[PurePosixPath, PurePosixPath | None]:
    for index, component in enumerate(relative.parts):
        if component.lower().endswith(".zip"):
            archive = PurePosixPath(*relative.parts[: index + 1])
            member_parts = relative.parts[index + 1 :]
            member = PurePosixPath(*member_parts) if member_parts else None
            return archive, member
    return relative, None


def _selected_zip_member(
    archive: zipfile.ZipFile, member: PurePosixPath
) -> zipfile.ZipInfo:
    selected = None
    names: set[str] = set()
    for info in archive.infolist():
        raw_name = info.filename[:-1] if info.is_dir() else info.filename
        relative = _canonical_relative_path(raw_name)
        name = relative.as_posix()
        if name in names:
            raise ValueError(f"duplicate Objaverse ZIP member: {name}")
        names.add(name)
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if not info.is_dir() and file_type not in (0, stat.S_IFREG):
            raise ValueError(f"unsafe Objaverse ZIP member type: {name}")
        if name == member.as_posix():
            if info.is_dir():
                raise ValueError(f"Objaverse ZIP member is a directory: {name}")
            selected = info
    if selected is None:
        raise ValueError(f"Objaverse ZIP member is missing: {member}")
    return selected


def _asset_digest(root: Path, relative: PurePosixPath) -> str:
    archive_relative, member = _split_archive_path(relative)
    digest = sha256()
    with _open_regular_file(root, archive_relative) as descriptor, os.fdopen(
        os.dup(descriptor), "rb"
    ) as source:
        if member is None:
            stream = source
            archive = None
        else:
            archive = zipfile.ZipFile(source, "r")
            info = _selected_zip_member(archive, member)
            stream = archive.open(info, "r")
        try:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        finally:
            if stream is not source:
                stream.close()
            if archive is not None:
                archive.close()
    return digest.hexdigest()


def _relative_download_path(root: Path, value: str) -> PurePosixPath:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        lexical = candidate.absolute().relative_to(root)
    except ValueError as error:
        raise ValueError(f"downloaded Objaverse asset escaped its root: {value}") from error
    return _canonical_relative_path(PurePosixPath(*lexical.parts).as_posix())


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='Data source to download annotations from (github, sketchfab)')


def get_metadata(source, **kwargs):
    if source == 'sketchfab':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    elif source == 'github':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
    else:
        raise ValueError(f"Invalid source: {source}")
    return metadata


def _existing_downloads(
    metadata: pd.DataFrame, output_dir: str
) -> tuple[list[dict[str, str]], pd.DataFrame]:
    root = Path(output_dir).resolve()
    existing: list[dict[str, str]] = []
    missing_indices: list[int] = []
    for index, row in metadata.iterrows():
        relative_value = row.get("local_path")
        if not isinstance(relative_value, str):
            missing_indices.append(index)
            continue
        try:
            relative = _canonical_relative_path(relative_value)
            actual = _asset_digest(root, relative)
            expected = _expected_digest(row)
        except (OSError, ValueError, zipfile.BadZipFile):
            missing_indices.append(index)
            continue
        if actual != expected:
            missing_indices.append(index)
            continue
        existing.append(
            {"sha256": row["sha256"], "local_path": relative.as_posix()}
        )
    return existing, metadata.loc[missing_indices]


def download(metadata: pd.DataFrame, output_dir: str, **kwargs) -> pd.DataFrame:
    max_workers = min(int(kwargs.get("max_workers", 8)), 8)
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    existing, pending = _existing_downloads(metadata, output_dir)
    if pending.empty:
        return pd.DataFrame(existing, columns=['sha256', 'local_path'])

    # download annotations
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(pending['sha256'].values)]
    
    # download and render objects
    file_paths = oxl.download_objects(
        annotations,
        download_dir=os.path.join(output_dir, "raw"),
        save_repo_format="zip",
        processes=max_workers,
    )
    
    downloaded = {record["sha256"]: record["local_path"] for record in existing}
    metadata = pending.set_index("file_identifier")
    for k, v in file_paths.items():
        row = metadata.loc[k]
        asset_sha = row["sha256"]
        try:
            relative = _relative_download_path(Path(output_dir).resolve(), v)
            verified = (
                _asset_digest(Path(output_dir).resolve(), relative)
                == _expected_digest(row)
            )
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            raise ValueError(
                f"downloaded Objaverse asset failed verification: {asset_sha}"
            ) from error
        if not verified:
            raise ValueError(
                f"downloaded Objaverse asset failed SHA-256 verification: {asset_sha}"
            )
        downloaded[asset_sha] = relative.as_posix()

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])


def _process_instance(args):
    """Worker function for ProcessPoolExecutor (must be top-level for pickling)"""
    metadatum, output_dir, func = args
    try:
        local_path = _canonical_relative_path(metadatum['local_path'])
        asset_sha = metadatum['sha256']
        root = Path(output_dir).resolve()
        archive_relative, member = _split_archive_path(local_path)
        with ExitStack() as stack:
            try:
                descriptor = stack.enter_context(
                    _open_regular_file(root, archive_relative)
                )
            except FileNotFoundError:
                # Later stages may need only the key and their prior output.
                file = root.joinpath(*local_path.parts)
                record = func(str(file), asset_sha)
            else:
                if member is not None:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        with os.fdopen(
                            os.dup(descriptor), "rb"
                        ) as source, zipfile.ZipFile(source, 'r') as zip_ref:
                            info = _selected_zip_member(zip_ref, member)
                            suffix = Path(member.name).suffix
                            file = Path(tmp_dir) / f"asset{suffix}"
                            with zip_ref.open(info, 'r') as packed, file.open(
                                'xb'
                            ) as target:
                                shutil.copyfileobj(
                                    packed, target, length=1024 * 1024
                                )
                        record = func(str(file), asset_sha)
                else:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        suffix = Path(archive_relative.name).suffix
                        file = Path(tmp_dir) / f"asset{suffix}"
                        file.symlink_to(f"/proc/self/fd/{descriptor}")
                        record = func(str(file), asset_sha)
        return record
    except Exception as e:
        print(f"Error processing object {metadatum.get('sha256', '?')}: {e}")
        return None


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects', log_interval=500, timeout=None) -> pd.DataFrame:
    print("================")
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
    from tqdm import tqdm
    
    # load metadata
    metadata = metadata.to_dict('records')

    max_workers = max_workers or os.cpu_count()
    records = []
    
    # Track processed/skipped counts
    total_processed = 0
    total_skipped = 0
    timeout_count = 0
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_instance, (m, output_dir, func)): m['sha256']
                for m in metadata
            }
            pbar = tqdm(as_completed(futures), total=len(futures), desc=desc)
            for future in pbar:
                sha256 = futures[future]
                try:
                    r = future.result(timeout=timeout)
                    if r is not None:
                        records.append(r)
                        # Update stats
                        if '_processed_count' in r:
                            total_processed += r['_processed_count']
                        if '_skipped_count' in r:
                            total_skipped += r['_skipped_count']
                        # Update progress bar display
                        pbar.set_postfix(processed=total_processed, skipped=total_skipped, timeout=timeout_count, refresh=False)
                except TimeoutError:
                    timeout_count += 1
                    print(f"Timeout processing object {sha256} (>{timeout}s)")
                    records.append({'sha256': sha256, 'error': f'Timeout (>{timeout}s)'})
                    pbar.set_postfix(processed=total_processed, skipped=total_skipped, timeout=timeout_count, refresh=False)
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
    except Exception as e:
        print(f"Error happened during processing: {e}")
    
    if timeout_count > 0:
        print(f"Total timeout: {timeout_count} objects")
        
    return pd.DataFrame.from_records(records)
