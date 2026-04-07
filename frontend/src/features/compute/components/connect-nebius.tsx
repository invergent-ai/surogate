// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Streamdown } from "streamdown";
import { code } from "@streamdown/code";
import { Field, FieldDescription, FieldLabel } from "@/components/ui/field";
import { GpuOfferPicker } from "@/components/gpu-offer-picker";
import { useAppStore } from "@/stores/app-store";
import { connectNebiusBackend } from "@/api/compute";

export function ConnectNebius({ onCancel }: { onCancel: () => void }) {
  const [serviceAccountId, setServiceAccountId] = useState("");
  const [publicKeyId, setPublicKeyId] = useState("");
  const [privateKeyPem, setPrivateKeyPem] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeProjectId = useAppStore((s) => s.activeProjectId);
  const fetchBackends = useAppStore((s) => s.fetchCloudBackends);
  const setBackendOffers = useAppStore((s) => s.setBackendOffers);
  const valid = serviceAccountId && publicKeyId && privateKeyPem && activeProjectId;

  const instructions = `
To connect your Nebius account to Surogate, please perform the following steps:

1. Log in to your Nebius account at [Nebius Console](https://console.nebius.com/)
2. Goto \`Administration\` → \`IAM\`
3. Click on the \`Create entity\` button and choose \`Service Account\`
4. Accept the default name and choose a project. Click \`Create and continue\`
5. In the next screen add it to the \`editors\` group and click \`Close\`,
6. Click on the newly created service account in the list of service accounts
7. Select the \`Authorized Keys\` tab and click the \`Upload authorized key\` button.
8. Upload the \`public.pem\` file generated in step 1 and click \`Upload\`.
9. Follow the instructions to create a public/private keypair. Click on \`Attach file\` and upload the generated \`public.pem\` file.
10. Set an \`Expiration date\` in the future and click \`Upload key\` to finish the process.
11. Copy the \`Public key ID\` from the newly created key and the \`Service Account ID\` and paste them below.
`

  async function handleConnect() {
    if (!valid) return;
    setSubmitting(true);
    setError(null);
    try {
      const pemContent = await privateKeyPem.text();
      const result = await connectNebiusBackend(activeProjectId, serviceAccountId, publicKeyId, pemContent);
      setBackendOffers(result.offers);
      await fetchBackends();
      setConnected(true);
    } catch (err: any) {
      setError(err.message || "Failed to connect Nebius backend");
    } finally {
      setSubmitting(false);
    }
  }

  return (
  <>
      <div className="pb-2 text-muted-foreground">
        <Streamdown mode="static" plugins={{ code }} controls={{ code: { copy: true, download: false } }}>
          {instructions}
        </Streamdown>
      </div>

      <div className="space-y-4">
        <Field>
          <FieldLabel htmlFor="service-account-id">Service Account ID</FieldLabel>
          <Input id="service-account-id" value={serviceAccountId} onChange={e => setServiceAccountId(e.target.value)} />
        </Field>
        <Field>
          <FieldLabel htmlFor="public-key-id">Public Key ID</FieldLabel>
          <Input id="public-key-id" value={publicKeyId} onChange={e => setPublicKeyId(e.target.value)} />
        </Field>
        <Field>
          <FieldLabel htmlFor="private-key-pem">Private key .pem file</FieldLabel>
          <Input id="private-key-pem" type="file" accept=".pem" onChange={e => setPrivateKeyPem(e.target.files?.[0] || null)} />
          <FieldDescription>Select the private key .pem file generated in step 9 above to upload.</FieldDescription>
        </Field>
      </div>

      {error && (
        <div className="text-sm text-destructive mt-3">{error}</div>
      )}

      {connected ? (
        <>
          <div className="mt-6 pt-4 border-t border-line space-y-3">
            <div className="text-sm text-green-500 font-medium">Nebius backend connected successfully. Available GPU instances:</div>
            <GpuOfferPicker backend="nebius" selectedOffer={null} onSelect={() => {}} />
          </div>
          <div className="flex gap-3 mt-4 pt-4 border-t border-line">
            <Button size="sm" onClick={onCancel}>Done</Button>
          </div>
        </>
      ) : (
        <div className="flex gap-3 mt-6 pt-4 border-t border-line">
          <Button size="sm" disabled={!valid || submitting} onClick={handleConnect}>
            {submitting ? "Verifying connection\u2026" : "Connect"}
          </Button>
          <Button variant="outline" size="sm" onClick={onCancel} disabled={submitting}>Cancel</Button>
        </div>
      )}
    </>
  );
}
