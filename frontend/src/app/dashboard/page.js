"use client"

import { Layout } from "antd";
import MapChart from "@/components/MapChart";
import WildfireInfo from "@/components/WildfireInfo";

const { Content } = Layout;

export default function Dashboard() {
  return (
    <Layout>
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
  );
}
