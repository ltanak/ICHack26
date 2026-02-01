import { MenuFoldOutlined, MenuUnfoldOutlined } from "@ant-design/icons";
import { Button } from "antd";

export default function Sidebar({ collapsed, setCollapsed }) {
    return (
        <div className={`flex flex-col items-end px-4 py-6 transition-all duration-300 mr-2`}>
            <Button 
                type="text" 
                    icon={
                    <span style={{ fontSize: 24 }}>
                    {collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                    </span>
                }
                onClick={() => setCollapsed(!collapsed)} 
            />
            <div className={`px-2 gap-y-6 flex flex-col items-start ${collapsed ? "opacity-0" : "opacity-100"} transition-all duration-300 mt-4 w-full`}>
                <h1 className="text-3xl"><strong>Wildfire Tracker</strong></h1>
                <div>
                    <h2 className="text-2xl"><strong>Information</strong></h2>
                    <ul className="">
                        <li className="text-lg">Wildfire Name: to add</li>
                        <li className="text-lg">Location: to add</li>
                        <li className="text-lg">Size: to add</li>
                        <li className="text-lg">Containment: to add</li>
                        <li className="text-lg">Start Date: to add</li>
                        <li className="text-lg">Cause: to add</li>
                    </ul>
                </div>
                <div>
                    <h2 className="text-2xl"><strong>Summary</strong></h2>
                    <p className="text-lg">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean porttitor magna eget tempus tristique. Suspendisse commodo arcu lacus, id iaculis elit aliquam non. Sed interdum dignissim turpis, vitae cursus neque volutpat eget. Aliquam vitae efficitur mauris. Integer ullamcorper, diam vitae pharetra tristique, lorem lorem lacinia diam, a cursus libero metus a felis. Sed luctus venenatis pellentesque. Donec sed nunc eget nunc placerat lobortis. Donec fringilla leo dolor, quis faucibus enim imperdiet sed. In fermentum libero ipsum, eget convallis eros gravida et. Aliquam ex massa, bibendum non rutrum vel, posuere mollis ante. Nam congue laoreet dictum. Nulla quis neque imperdiet, fringilla lacus ac, iaculis mi. Nunc laoreet lacinia feugiat. Aliquam nec rhoncus nisi, a malesuada velit.</p>
                    <img
                    src="http://127.0.0.1:5000/satellite"
                    alt="Satellite view"
                    className="w-full h-auto"
                    />
                </div>
            </div>
        </div>
    )
}