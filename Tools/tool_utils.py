import os
import unicodedata


BINARY_EXTENSIONS = {
    'pdf', 'png', 'jpg', 'jpeg', 'gif', 'zip', 'tar', 'gz',
    'exe', 'bin', 'so', 'dll', 'pyc', 'mp3', 'mp4', 'mov'
}
BLOCKED_DEVICE_PATHS = {
  '/dev/zero',
  '/dev/random',
  '/dev/urandom',
  '/dev/full',
  '/dev/stdin',
  '/dev/tty',
  '/dev/console',
  '/dev/stdout',
  '/dev/stderr',
  '/dev/fd/0',
  '/dev/fd/1',
  '/dev/fd/2',
}

def tool_dispatch(tool_name: str):
    pass


def expandPath(path: str, baseDir: str | None = None) -> str:
    """
    Expands a path that may contain tilde notation (~) to an absolute path.
    Linux only.
    
    Examples:
    - expandPath('~') → '/home/user'
    - expandPath('~/Documents') → '/home/user/Documents'
    - expandPath('./src', '/project') → '/project/src'
    """

    actualBaseDir = _normalize_path(baseDir or os.getcwd())
    
    trimmedPath = path.strip()
    if not trimmedPath:
        return actualBaseDir
    
    # Handle home directory notation
    if trimmedPath == '~':
        return _normalize_path(os.path.expanduser('~'))
    
    if trimmedPath.startswith('~/'):
        return _normalize_path(os.path.expanduser(trimmedPath))
    
    # Handle absolute paths
    if os.path.isabs(trimmedPath):
        return _normalize_path(trimmedPath)
    
    # Handle relative paths (resolve against baseDir)
    resolved = os.path.join(actualBaseDir, trimmedPath)
    return _normalize_path(resolved)


def _normalize_path(path: str) -> str:
    """Normalize path and apply Unicode NFC normalization"""
    normalized = os.path.normpath(path)
    return unicodedata.normalize('NFC', normalized)