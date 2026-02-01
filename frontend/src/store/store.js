import { configureStore } from '@reduxjs/toolkit';
import { api } from './api';
import pointsReducer from './features/points/pointsSlice';

export const store = configureStore({
  reducer: {
    [api.reducerPath]: api.reducer, // adds the RTK Query slice
    points: pointsReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(api.middleware), // adds RTK Query middleware
});