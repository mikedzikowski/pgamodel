"""User registration, login, and profile routes."""
from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...db import crud
from ..auth import create_access_token, hash_password, verify_password
from ..deps import get_db, get_current_user
from ...db.models import User

router = APIRouter(prefix="/users", tags=["users"])


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserProfile(BaseModel):
    id: str
    email: str
    username: str
    tier: str
    is_active: bool


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if crud.get_user_by_email(db, body.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    if crud.get_user_by_username(db, body.username):
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed = hash_password(body.password)
    user = crud.create_user(db, body.email, body.username, hashed)
    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, body.email)
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserProfile)
def get_profile(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sub = crud.get_subscription(db, user.id)
    tier = sub.tier if sub and sub.is_active else "free"
    return UserProfile(
        id=user.id,
        email=user.email,
        username=user.username,
        tier=tier,
        is_active=user.is_active,
    )
