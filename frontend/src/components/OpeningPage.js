"use client";

export default function OpeningPage({ onEnterClick }) {
  return (
    <div className="min-h-screen bg-gradient-to-b from-orange-900 via-red-800 to-orange-700 flex items-center justify-center px-4">
      <div className="text-center max-w-2xl">
        <div className="text-8xl mb-6">🔥</div>
        <h1 className="text-6xl font-bold text-white mb-4">Wildfire</h1>
        <h2 className="text-2xl text-orange-200 mb-8 font-semibold">Model. Mitigate. Predict.</h2>
        <p className="text-xl text-gray-100 mb-12 leading-relaxed">
          Predict wildfires with an advanced simulation model.
        </p>
        <button
          onClick={onEnterClick}
          className="px-8 py-4 bg-white text-orange-700 font-bold text-lg rounded-lg hover:bg-gray-100 transition-colors duration-200 shadow-lg"
        >
          Enter App
        </button>
      </div>
    </div>
  );
}
