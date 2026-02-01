import { NextResponse } from "next/server";

export async function GET(request, { params }) {
    const { searchParams } = new URL(request.url);
    const { point } = await params;

    const response = await fetch(`${process.env.API_URL}/summary/${encodeURIComponent(point)}`);
    if (!response.ok) {
        return NextResponse.json({ error: "Failed to fetch point summary" }, { status: 500 });
    }
    const data = await response.json();
    return NextResponse.json(data);
}