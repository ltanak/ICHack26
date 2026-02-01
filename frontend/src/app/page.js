"use client"

import { Divider, Layout } from "antd";
import { useState } from "react";
import MapChart from "../components/MapChart";
import Sidebar from "@/components/Sidebar";
import WildfireInfo from "@/components/WildfireInfo";

const { Header, Content, Footer, Sider } = Layout;

export default function Home() {
  const [collapsed, setCollapsed] = useState(true);

  return (
    <Layout>
      {/* <Sider trigger={null} width={650} collapsible collapsed={collapsed} onCollapse={(value) => setCollapsed(value)} className="!bg-stone-200 shadow-2xl!">
        <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />
      </Sider> */}
      <Layout>
        <Content className="bg-slate-200! flex h-screen flex-col">
          <div className="overflow-hidden ">
            <MapChart />
          </div>
          <Content className="bg-slate-200! flex">
            <WildfireInfo />
          </Content>
        </Content>
      </Layout>
    </Layout>
  );
}
