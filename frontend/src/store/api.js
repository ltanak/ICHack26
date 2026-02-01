import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";

export const api = createApi({
  reducerPath: "api",
  baseQuery: fetchBaseQuery({ baseUrl: "/api" }),
  endpoints: (builder) => ({
    getPoints: builder.query({
      query: () => "/points",
    }),
    getPointSummary: builder.query({
      query: (point) => `/${encodeURIComponent(point)}/summary`,
    }),
  }),
});

export const { useGetPointsQuery, useGetPointSummaryQuery } = api;