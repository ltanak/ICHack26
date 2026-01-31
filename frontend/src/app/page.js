"use client"

import { Button, Layout, Menu } from "antd";
import { useState } from "react";
import MapChart from "../components/MapChart";
import { MenuFoldOutlined, MenuUnfoldOutlined } from "@ant-design/icons";

const { Header, Content, Footer, Sider } = Layout;

export default function Home() {
  const [collapsed, setCollapsed] = useState(true);

  return (
    <Layout>
      <Sider trigger={null}collapsible collapsed={collapsed} onCollapse={(value) => setCollapsed(value)} className="!bg-stone-200">
        <div className={`flex flex-col items-end px-4 py-6 transition-all duration-300 mr-2`}>
          <Button type="text" icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />} onClick={() => setCollapsed(!collapsed)} />
        </div>
      </Sider>
      <Layout>
        <Content>
          <div className="overflow-hidden">
            <MapChart />
          </div>
        </Content>
      </Layout>
    </Layout>
  );
}
