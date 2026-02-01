import { NextResponse } from "next/server";

export async function GET(request) {
    const response = await fetch(`${process.env.API_URL}/points`);
    if (!response.ok) {
        return NextResponse.json({ error: "Failed to fetch points" }, { status: 500 });
    }
    const data = await response.json();
    return NextResponse.json(data);
}