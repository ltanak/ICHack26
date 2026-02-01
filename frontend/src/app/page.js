"use client"

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import OpeningPage from "@/components/OpeningPage";
import dynamic from "next/dynamic";

// Preload dashboard components in the background
const DashboardPreload = dynamic(() => import("@/app/dashboard/page"), {
  ssr: false,
  loading: () => null,
});

export default function Home() {
  const router = useRouter();

  // Prefetch the dashboard page in the background
  useEffect(() => {
    router.prefetch("/dashboard");
  }, [router]);

  const handleEnterClick = () => {
    router.push("/dashboard");
  };

  return (
    <>
      <OpeningPage onEnterClick={handleEnterClick} />
      <div style={{ display: "none" }}>
        <DashboardPreload />
      </div>
    </>
  );
}
