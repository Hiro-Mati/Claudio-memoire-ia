# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for implicit ACL MANAGE short-circuit by role."""

from openviking.server.identity import RequestContext, Role
from openviking.storage.acl import has_implicit_manage
from openviking_cli.session.user_id import UserIdentifier


def _ctx(role: Role) -> RequestContext:
    return RequestContext(user=UserIdentifier(account_id="acct", user_id="u1"), role=role)


ACL_URI = "viking://resources/proj/doc.md"
NON_ACL_URI = "viking://upload/2026082913/abc"


def test_root_and_admin_have_implicit_manage_on_acl_uri():
    # Both privileged roles skip per-URI ACL resolution on ACL-scoped URIs.
    assert has_implicit_manage(_ctx(Role.ROOT), ACL_URI) is True
    assert has_implicit_manage(_ctx(Role.ADMIN), ACL_URI) is True


def test_user_has_no_implicit_manage():
    assert has_implicit_manage(_ctx(Role.USER), ACL_URI) is False


def test_no_implicit_manage_on_non_acl_uri_even_for_root():
    # Non-ACL URIs (e.g. shared uploads) are not ACL-scoped, so the implicit
    # MANAGE short-circuit never applies regardless of role.
    assert has_implicit_manage(_ctx(Role.ROOT), NON_ACL_URI) is False
    assert has_implicit_manage(_ctx(Role.ADMIN), NON_ACL_URI) is False
