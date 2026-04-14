// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import type { SyntheticEvent } from "react";
import { cn } from "@/utils/cn";
import { useNavigate } from "@tanstack/react-router";
import { getPostAuthRoute, storeAuthTokens } from "./session";

type AuthMethod = "credentials" | "sso";

const AUTH_METHODS: { id: AuthMethod; label: string; disabled?: boolean }[] = [
  { id: "credentials", label: "Credentials" },
  { id: "sso", label: "SSO / OIDC", disabled: true },
];
const SSO_PROVIDERS = ["Okta", "Azure AD", "Auth0", "Keycloak"] as const;

type TokenResponse = {
  access_token: string;
  refresh_token: string;
  must_change_password: boolean;
};

async function loginWithPassword(
  username: string,
  password: string,
): Promise<TokenResponse> {
  const response = await fetch("/api/auth/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      username: username.trim(),
      password,
    }),
  });

  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(errorPayload?.detail ?? "Login failed.");
  }

  return (await response.json()) as TokenResponse;
}

export function LoginForm() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [authMethod, setAuthMethod] = useState<AuthMethod>("credentials");
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [loginError, setLoginError] = useState<string | null>(null);

  const handleLogin = async (e: SyntheticEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setLoginError("Please enter both username and password");
      return;
    }
    setLoginError(null);
    setIsLoading(true);


    try {
      const token = await loginWithPassword(email, password);
      if (token.must_change_password) {
        console.warn("User must change password - redirecting to change password page NOT IMPLEMENTED");
        return;
      }
      storeAuthTokens(
        token.access_token,
        token.refresh_token,
        token.must_change_password,
      );
      navigate({ to: getPostAuthRoute() });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Auth failed.";
      setLoginError(msg);
    }  finally {
      setIsLoading(false);
    }
  };

  const clearError = () => setLoginError(null);

  const inputCls = (field: string, hasError = false) =>
    cn(
      "w-full py-[11px] px-3.5 rounded-lg border font-mono text-sm text-foreground bg-input outline-none transition-all duration-200",
      focusedField === field
        ? "border-primary/25 bg-input-focus shadow-[0_0_0_3px_rgba(245,158,11,0.06)]"
        : hasError
          ? "border-destructive/20"
          : "border-border",
    );

  return (
    <>
      {/* heading */}
      <div className="mb-8">
        <h3 className="font-display text-5xl font-bold text-foreground tracking-tight mb-1.5">
          Sign in
        </h3>
        <p className="font-display text-xl text-muted-foreground">
          Access your workspace
        </p>
      </div>

      {/* auth method tabs */}
      <div className="flex mb-6 bg-muted rounded-lg p-[3px] border border-line">
        {AUTH_METHODS.map((m) => (
          <button
            key={m.id}
            type="button"
            disabled={m.disabled}
            onClick={() => {
              setAuthMethod(m.id);
              clearError();
            }}
            className={cn(
              "flex-1 py-2 rounded-md border-none font-display text-sm transition-all duration-150",
              m.disabled
                ? "cursor-not-allowed opacity-40 text-muted-foreground"
                : "cursor-pointer",
              !m.disabled && authMethod === m.id
                ? "bg-accent text-accent-foreground font-semibold"
                : !m.disabled && "bg-transparent text-muted-foreground",
            )}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* ─── EMAIL / PASSWORD FORM ─── */}
      {authMethod === "credentials" && (
        <form
          onSubmit={handleLogin}
          className="animate-[fade-in_0.2s_ease]"
        >
          <div className="mb-4">
            <label className="block font-display text-sm text-subtle font-medium mb-1.5">
              Username
            </label>
            <input
              type="text"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value);
                clearError();
              }}
              onFocus={() => setFocusedField("username")}
              onBlur={() => setFocusedField(null)}
              placeholder="your username"
              className={inputCls("username", !!loginError && !email)}
            />
          </div>

          <div className="mb-2">
            <label className="block font-display text-sm text-subtle font-medium mb-1.5">
              Password
            </label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  clearError();
                }}
                onFocus={() => setFocusedField("password")}
                onBlur={() => setFocusedField(null)}
                placeholder="••••••••••••"
                className={cn(inputCls("password"), "pr-11")}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 bg-transparent border-none text-faint cursor-pointer text-sm font-display"
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
          </div>

          {/* remember + forgot */}
          <div className="flex items-center justify-between mb-6">
            <label className="flex items-center gap-1.5 cursor-pointer">
              <div
                onClick={() => setRememberMe(!rememberMe)}
                onKeyDown={(e) => {
                  if (e.key === " " || e.key === "Enter")
                    setRememberMe(!rememberMe);
                }}
                role="checkbox"
                aria-checked={rememberMe}
                tabIndex={0}
                className={cn(
                  "w-4 h-4 rounded-sm border flex items-center justify-center cursor-pointer transition-all duration-150",
                  rememberMe
                    ? "border-primary/25 bg-primary/10"
                    : "border-border bg-transparent",
                )}
              >
                {rememberMe && (
                  <span className="text-sm text-primary">✓</span>
                )}
              </div>
              <span className="font-display text-sm text-muted-foreground">
                Remember me
              </span>
            </label>
            <button
              type="button"
              className="bg-transparent border-none text-primary cursor-pointer text-sm font-display font-medium"
            >
              Forgot password?
            </button>
          </div>

          {/* error */}
          {loginError && (
            <div className="mb-4 px-3.5 py-2.5 rounded-lg bg-destructive/5 border border-destructive/15 text-sm text-destructive font-display animate-[fade-in_0.2s_ease]">
              {loginError}
            </div>
          )}

          {/* submit */}
          <button
            type="submit"
            disabled={isLoading}
            className={cn(
              "w-full py-[13px] rounded-lg border-none font-display text-sm font-bold tracking-tight flex items-center justify-center gap-2 transition-all duration-200",
              isLoading
                ? "bg-amber-600/50 cursor-not-allowed text-primary-foreground"
                : "bg-gradient-to-br from-amber-500 to-amber-600 cursor-pointer text-primary-foreground shadow-[0_4px_24px_rgba(245,158,11,0.15)]",
            )}
          >
            {isLoading ? (
              <>
                <span className="w-4 h-4 border-2 border-primary-foreground/25 border-t-primary-foreground rounded-full animate-spinner" />
                Signing in...
              </>
            ) : (
              "Sign in"
            )}
          </button>
        </form>
      )}

      {/* ─── SSO / OIDC FORM ─── */}
      {authMethod === "sso" && (
        <div className="animate-[fade-in_0.2s_ease]">
          <div className="mb-4">
            <label className="block font-display text-sm text-subtle font-medium mb-1.5">
              Organization Domain
            </label>
            <input
              type="text"
              placeholder="company.com"
              onFocus={() => setFocusedField("domain")}
              onBlur={() => setFocusedField(null)}
              className={inputCls("domain")}
            />
          </div>
          <button
            type="button"
            className="w-full py-[13px] rounded-lg border-none cursor-pointer bg-gradient-to-br from-amber-500 to-amber-600 text-primary-foreground font-display text-sm font-bold shadow-[0_4px_24px_rgba(245,158,11,0.15)]"
          >
            Continue with SSO
          </button>
          <div className="mt-4 flex flex-col gap-2">
            <p className="font-display text-sm text-faint text-center">
              Supported providers:
            </p>
            <div className="flex justify-center gap-2">
              {SSO_PROVIDERS.map((p) => (
                <span
                  key={p}
                  className="font-display text-[11px] px-2 py-[3px] rounded bg-muted border border-border text-muted-foreground"
                >
                  {p}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
