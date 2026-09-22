"""Esquemas Pydantic del modulo de autenticacion (users, roles, permissions)."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------- Permission


class PermissionBase(BaseModel):
    code: str = Field(max_length=100)
    description: str | None = None


class PermissionCreate(PermissionBase):
    pass


class PermissionUpdate(BaseModel):
    code: str | None = Field(default=None, max_length=100)
    description: str | None = None


class PermissionRead(PermissionBase):
    model_config = ConfigDict(from_attributes=True)

    id: int


# -------------------------------------------------------------------------- Role


class RoleBase(BaseModel):
    name: str = Field(max_length=50)
    description: str | None = None


class RoleCreate(RoleBase):
    permission_ids: list[int] = Field(default_factory=list)


class RoleUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=50)
    description: str | None = None
    permission_ids: list[int] | None = None


class RoleRead(RoleBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    permissions: list[PermissionRead] = Field(default_factory=list)


# -------------------------------------------------------------------------- User


class UserBase(BaseModel):
    email: str = Field(max_length=320)
    username: str = Field(max_length=150)
    full_name: str | None = None
    is_active: bool = True
    is_superuser: bool = False


class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)
    role_ids: list[int] = Field(default_factory=list)


class UserUpdate(BaseModel):
    email: str | None = Field(default=None, max_length=320)
    username: str | None = Field(default=None, max_length=150)
    full_name: str | None = None
    is_active: bool | None = None
    is_superuser: bool | None = None
    password: str | None = Field(default=None, min_length=8, max_length=128)
    role_ids: list[int] | None = None


class UserRead(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    last_login_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    roles: list[RoleRead] = Field(default_factory=list)


# --------------------------------------------------------------- Alta de cuenta


class RegisterRequest(BaseModel):
    email: str = Field(max_length=320, examples=["ana@ejemplo.org"])
    username: str = Field(min_length=3, max_length=150, examples=["ana"])
    password: str = Field(min_length=8, max_length=128, examples=["cambia-esta-clave"])
    full_name: str | None = Field(default=None, examples=["Ana Perez"])


class RoleAssignment(BaseModel):
    role_ids: list[int] = Field(default_factory=list)


class UserAdminUpdate(BaseModel):
    full_name: str | None = None
    is_active: bool | None = None
    is_superuser: bool | None = None
    password: str | None = Field(default=None, min_length=8, max_length=128)
    role_ids: list[int] | None = None
