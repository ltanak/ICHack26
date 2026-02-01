"use client"

import { Divider, Layout } from "antd";
import { useState, useRef } from "react";
import MapChart from "../components/MapChart";
import Sidebar from "@/components/Sidebar";
import WildfireSimulation from "@/components/WildfireSimulation";
import WildfireInfo from "@/components/WildfireInfo";
import OpeningPage from "@/components/OpeningPage";

const { Header, Content, Footer, Sider } = Layout;

export default function Home() {
  const [collapsed, setCollapsed] = useState(true);
  const appRef = useRef(null);

  const handleEnterClick = () => {
    appRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <>
      <OpeningPage onEnterClick={handleEnterClick} />
      <div ref={appRef}>
        <Layout>
          {/* <Sider trigger={null} width={650} collapsible collapsed={collapsed} onCollapse={(value) => setCollapsed(value)} className="!bg-stone-200 shadow-2xl!">
            <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />
          </Sider> */}
          <Layout>
            <Content className="bg-slate-200! flex min-h-screen flex-col">
              <div className="overflow-hidden">
                <MapChart />
              </div>
              <Content className="bg-slate-200! flex">
                <WildfireInfo />
              </Content>
            </Content>
          </Layout>
        </Layout>
      </div>
    </>
  );
}
