import io
import zipfile
import requests
from pathlib import PurePosixPath
from urllib.parse import urlparse
from typing import List, Dict, Any
from src.connectors.base import BaseConnector
from src.core.logger import setup_logger

logger = setup_logger(__name__)

# Supported source code, documentation, and configuration extensions
SUPPORTED_EXTENSIONS = {
    # Source code
    ".py", ".java", ".js", ".ts", ".jsx", ".tsx",
    ".c", ".cpp", ".h", ".cs", ".go", ".rb", ".rs",
    ".swift", ".kt", ".php", ".scala", ".sh", ".bash",
    # Documentation
    ".md", ".rst", ".txt",
    # Configuration
    ".json", ".yaml", ".yml", ".xml", ".ini",
    ".conf", ".toml", ".cfg", ".env",
    # Web
    ".html", ".htm", ".css",
    # Special named files (no extension)
    "dockerfile", ".gitignore", ".gitattributes",
    "makefile", "procfile", "requirements",
}

# Folders to exclude from traversal
EXCLUDED_FOLDERS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "env", "dist", "build", ".idea", ".vscode", "target",
    "bin", "obj", ".gradle", ".mvn", "vendor",
}

# Binary file extensions to skip
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".tiff", ".tif",
    ".svg", ".webp", ".mp4", ".mp3", ".wav", ".avi", ".mov",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
    ".class", ".pyc", ".pyo", ".o", ".a",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".lock", ".sum",
}


class GitHubConnector(BaseConnector):
    def __init__(self, repo_url: str, branch: str = "main", token: str = None):
        """
        Args:
            repo_url: Full GitHub URL (https://github.com/owner/repo) or shorthand (owner/repo).
            branch:   Branch name to download (default: main).
            token:    Optional GitHub Personal Access Token for private repos / higher rate limits.
        """
        self.repo_url = repo_url.strip().rstrip("/")
        self.branch = branch.strip()
        self.token = token
        self.owner, self.repo = self._parse_repo_url()

    def _parse_repo_url(self):
        """Parse owner and repo name from URL or shorthand."""
        url = self.repo_url
        if url.startswith("http"):
            parts = urlparse(url).path.strip("/").split("/")
            if len(parts) < 2:
                raise ValueError(f"Invalid GitHub repository URL: {url}")
            return parts[0], parts[1]
        elif "/" in url:
            parts = url.split("/")
            return parts[0], parts[1]
        else:
            raise ValueError(f"Invalid repository format: {url}. Use 'owner/repo' or full GitHub URL.")

    def fetch_documents(self) -> List[Dict[str, Any]]:
        """
        Downloads the repository as a ZIP archive and extracts supported files.
        Returns list of document dicts with raw_data, source, extension, and metadata.
        """
        zip_url = f"https://github.com/{self.owner}/{self.repo}/archive/refs/heads/{self.branch}.zip"
        logger.info(f"Downloading repository archive: {zip_url}")

        headers = {"User-Agent": "DocuMind-Crawler/1.0"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"

        try:
            response = requests.get(zip_url, headers=headers, timeout=60)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to download repository archive. Status: {response.status_code}. "
                    f"Check that the repository is public and the branch '{self.branch}' exists."
                )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error downloading repository: {e}")

        logger.info(f"Archive downloaded ({len(response.content) / 1024:.1f} KB). Extracting files...")

        documents = []
        zip_root_prefix = f"{self.repo}-{self.branch}/"

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            for zip_entry in zf.infolist():
                # Skip directories
                if zip_entry.filename.endswith("/"):
                    continue

                # Strip the root prefix (e.g. "repo-main/") to get relative path
                rel_path = zip_entry.filename
                if rel_path.startswith(zip_root_prefix):
                    rel_path = rel_path[len(zip_root_prefix):]

                # Skip if inside an excluded folder
                path_parts = PurePosixPath(rel_path).parts
                if any(part in EXCLUDED_FOLDERS for part in path_parts):
                    continue

                # Determine extension
                file_name = path_parts[-1] if path_parts else rel_path
                ext = PurePosixPath(file_name).suffix.lower()
                base_name_lower = file_name.lower()

                # Skip binary files
                if ext in BINARY_EXTENSIONS:
                    continue

                # Check if supported: by extension or by special filename
                is_supported = (
                    ext in SUPPORTED_EXTENSIONS or
                    base_name_lower in SUPPORTED_EXTENSIONS or
                    ext == ""  # extensionless files (Makefile, Procfile, etc.)
                )
                if not is_supported:
                    continue

                try:
                    raw_data = zf.read(zip_entry.filename)
                    # Skip empty files
                    if not raw_data.strip():
                        continue
                    # Try decoding to confirm it's text-based; skip if not
                    raw_data.decode("utf-8", errors="strict")
                except (UnicodeDecodeError, KeyError):
                    continue

                # Build folder hierarchy
                folder_parts = list(path_parts[:-1])  # all except filename
                folder_hierarchy_str = "/".join(folder_parts) if folder_parts else ""

                # Build direct GitHub file URL for citation
                source_url = f"https://github.com/{self.owner}/{self.repo}/blob/{self.branch}/{rel_path}"

                documents.append({
                    "raw_data": raw_data,
                    "source": source_url,
                    "extension": ext if ext else ".txt",
                    "metadata": {
                        "repository": f"{self.owner}/{self.repo}",
                        "branch": self.branch,
                        "file_path": rel_path,
                        "file_name": file_name,
                        "folder_hierarchy": folder_parts,
                        "folder_hierarchy_str": folder_hierarchy_str,
                    }
                })

        logger.info(f"Extracted {len(documents)} supported text files from repository {self.owner}/{self.repo}.")
        return documents
