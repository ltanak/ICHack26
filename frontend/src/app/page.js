"use client"

import { useRouter } from "next/navigation";
import OpeningPage from "@/components/OpeningPage";

export default function Home() {
  const router = useRouter();

  const handleEnterClick = () => {
    router.push("/dashboard");
  };

  return <OpeningPage onEnterClick={handleEnterClick} />;
}
