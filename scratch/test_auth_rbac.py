import sys
from pathlib import Path
import os
import uuid
import time

# Add project root and src folder to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Configure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.core.auth import (
    register_user, authenticate_user, create_access_token, decode_access_token, hash_password
)
from src.core.chat_history import save_document_meta, load_document_meta

def test_auth_and_rbac():
    print("=" * 80)
    print("STARTING AUTHENTICATION, JWT & RBAC VERIFICATION SUITE")
    print("=" * 80)
    
    # 1. Password Hashing & Salt Test
    print("→ Testing SHA-256 password hashing with salt...")
    hash1, salt1 = hash_password("secret_pass_123")
    hash2, salt2 = hash_password("secret_pass_123", salt1)
    assert hash1 == hash2, "Password hashing should be deterministic given the same salt."
    print("✓ Password hashing & salt verified.\n")
    
    # 2. Default Seed Accounts Test
    print("→ Testing default seed accounts ('admin' and 'user')...")
    admin_auth = authenticate_user("admin", "admin123")
    user_auth = authenticate_user("user", "user123")
    
    assert admin_auth is not None, "Default admin account authentication failed."
    assert admin_auth["role"] == "admin", "Admin account should have role 'admin'."
    assert user_auth is not None, "Default user account authentication failed."
    assert user_auth["role"] == "user", "User account should have role 'user'."
    print(f"✓ Default accounts verified: Admin ({admin_auth['user_id'][:8]}...), User ({user_auth['user_id'][:8]}...)\n")

    # 3. User Registration Test
    test_uname = f"test_user_{uuid.uuid4().hex[:6]}"
    print(f"→ Registering new test user: '{test_uname}'...")
    ok, msg = register_user(test_uname, "password123", role="user")
    assert ok, f"User registration failed: {msg}"
    
    auth_res = authenticate_user(test_uname, "password123")
    assert auth_res is not None, "Failed to authenticate newly registered user."
    print(f"✓ Registration & authentication verified for user '{test_uname}'.\n")

    # 4. JWT Token Encoding & Decoding Test
    print("→ Testing JWT Token creation and verification...")
    token = create_access_token(auth_res, expires_delta_hours=1)
    decoded = decode_access_token(token)
    
    assert decoded is not None, "JWT token decoding failed."
    assert decoded["username"] == test_uname, "JWT payload username mismatch."
    assert decoded["role"] == "user", "JWT payload role mismatch."
    print("✓ JWT Token generation and verification successful.\n")

    # 5. RBAC Document Scoping & Isolation Test
    print("→ Testing RBAC Document Isolation...")
    user_a_id = f"user_a_{uuid.uuid4().hex[:6]}"
    user_b_id = f"user_b_{uuid.uuid4().hex[:6]}"
    
    doc_a = f"confidential_doc_a_{uuid.uuid4().hex[:4]}.pdf"
    doc_b = f"confidential_doc_b_{uuid.uuid4().hex[:4]}.pdf"
    
    save_document_meta(user_id=user_a_id, filename=doc_a, chunk_count=5)
    save_document_meta(user_id=user_b_id, filename=doc_b, chunk_count=8)
    
    # User A document check
    user_a_docs = [d["filename"] for d in load_document_meta(user_a_id)]
    assert doc_a in user_a_docs, "User A should see doc_a."
    assert doc_b not in user_a_docs, "User A MUST NOT see User B's doc_b."
    
    # User B document check
    user_b_docs = [d["filename"] for d in load_document_meta(user_b_id)]
    assert doc_b in user_b_docs, "User B should see doc_b."
    assert doc_a not in user_b_docs, "User B MUST NOT see User A's doc_a."
    
    # Admin document check (all)
    admin_docs = [d["filename"] for d in load_document_meta("all")]
    assert doc_a in admin_docs and doc_b in admin_docs, "Admin should see documents across all users."
    
    print("✓ RBAC Document Scoping verified: Standard users isolated, Admin has full visibility.")

    print("=" * 80)
    print("AUTHENTICATION & RBAC SUITE COMPLETED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    test_auth_and_rbac()
