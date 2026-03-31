// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Button } from "@/components/ui/button";
import { CURRENT_USER, PROJECTS } from "./settings-data";

const userProjects = PROJECTS.filter((p) =>
  p.members.some((m) => m.email === CURRENT_USER.email),
);

const PROFILE_FIELDS = [
  { label: "Full Name", value: CURRENT_USER.name },
  { label: "Email", value: CURRENT_USER.email },
  { label: "Role", value: CURRENT_USER.role },
  { label: "Default Project", value: CURRENT_USER.defaultProject },
  { label: "Member Since", value: CURRENT_USER.createdAt },
  { label: "Last Login", value: CURRENT_USER.lastLogin },
];

export function ProfileTab() {
  return (
    <div>
      <h2 className="font-display text-[15px] font-bold text-foreground mb-5">
        Profile
      </h2>

      {/* Avatar + info */}
      <div className="flex items-center gap-4 mb-6">
        <div className="w-14 h-14 rounded-full bg-linear-to-br from-blue-500 to-violet-500 flex items-center justify-center text-lg font-bold text-white shrink-0">
          {CURRENT_USER.avatar}
        </div>
        <div>
          <div className="font-display text-base font-bold text-foreground">
            {CURRENT_USER.name}
          </div>
          <div className="text-[11px] text-muted-foreground">
            {CURRENT_USER.role} · {CURRENT_USER.email}
          </div>
          <div className="flex gap-1 mt-1">
            {userProjects.map((p) => (
              <span
                key={p.id}
                className="text-[8px] px-1.5 py-[2px] rounded bg-accent"
                style={{ color: p.color }}
              >
                {p.name}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Details card */}
      <div className="bg-card border border-line rounded-lg overflow-hidden">
        {PROFILE_FIELDS.map((f) => (
          <div
            key={f.label}
            className="px-4 py-2.5 border-b border-line last:border-b-0 flex justify-between items-center"
          >
            <span className="text-[11px] text-muted-foreground">
              {f.label}
            </span>
            <span className="text-[11px] text-foreground/70">{f.value}</span>
          </div>
        ))}
      </div>

      {/* Actions */}
      <div className="mt-4 flex gap-2">
        <Button variant="outline" size="sm">
          Edit Profile
        </Button>
        <Button variant="outline" size="sm">
          Change Password
        </Button>
      </div>
    </div>
  );
}
