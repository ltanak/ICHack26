import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";

export const api = createApi({
  reducerPath: "api",
  baseQuery: fetchBaseQuery({ baseUrl: "/api" }),
  endpoints: (builder) => ({
    getPoints: builder.query({
      query: () => "/points",
    }),
  }),
});

export const { useGetPointsQuery } = api;