import { createSlice } from "@reduxjs/toolkit";

const initialState = {
    selectedPoint: null,
}

const pointsSlice = createSlice({
    name: "points",
    initialState,
    reducers: {
        setSelectedPoint(state, action) {
            state.selectedPoint = action.payload;
        },
        clearSelectedPoint(state) {
            state.selectedPoint = null;
        }
    },
});

export const { setSelectedPoint, clearSelectedPoint } = pointsSlice.actions;
export default pointsSlice.reducer;