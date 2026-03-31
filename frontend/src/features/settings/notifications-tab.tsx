// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Switch } from "@/components/ui/switch";
import { NOTIFICATION_CATEGORIES, type NotificationCategory } from "./settings-data";

function NotificationSection({ category }: { category: NotificationCategory }) {
  const [items, setItems] = useState(category.items);

  const toggle = (index: number, field: "email" | "inApp") => {
    setItems((prev) =>
      prev.map((item, i) =>
        i === index ? { ...item, [field]: !item[field] } : item,
      ),
    );
  };

  return (
    <div className="mb-5">
      <div className="font-display text-[11px] font-semibold text-foreground mb-2">
        {category.category}
      </div>
      <div className="bg-card border border-line rounded-lg overflow-hidden">
        {/* Header */}
        <div className="grid grid-cols-[1fr_60px_60px] px-4 py-1.5 border-b border-line">
          <span />
          <span className="text-[8px] text-faint uppercase tracking-[0.1em] text-center font-display">
            Email
          </span>
          <span className="text-[8px] text-faint uppercase tracking-[0.1em] text-center font-display">
            In-App
          </span>
        </div>
        {/* Rows */}
        {items.map((item, i) => (
          <div
            key={item.label}
            className="grid grid-cols-[1fr_60px_60px] items-center px-4 py-2 border-b border-line last:border-b-0"
          >
            <span className="text-[11px] text-foreground/70">{item.label}</span>
            <div className="flex justify-center">
              <Switch
                size="sm"
                checked={item.email}
                onCheckedChange={() => toggle(i, "email")}
              />
            </div>
            <div className="flex justify-center">
              <Switch
                size="sm"
                checked={item.inApp}
                onCheckedChange={() => toggle(i, "inApp")}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function NotificationsTab() {
  return (
    <div>
      <h2 className="font-display text-[15px] font-bold text-foreground mb-5">
        Notification Preferences
      </h2>
      {NOTIFICATION_CATEGORIES.map((cat) => (
        <NotificationSection key={cat.category} category={cat} />
      ))}
    </div>
  );
}
