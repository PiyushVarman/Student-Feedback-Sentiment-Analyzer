import { useState } from 'react'
import logo from './assets/vitlogo.png'
import './App.css'

function App() {
  const [score, setScore] = useState("");
  const [feedback, setFeedback] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!feedback.trim()) return;

    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          score: Number(score),
          feedback: feedback,
        }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
    }

    setLoading(false);
  };

  return (
    <>
      <div className='w-[40vw] bg-white rounded-3xl drop-shadow-2xl *:transition-all *:duration-100 ease-in-out'>
        <img src={logo} className='m-auto w-[20vw] pt-[2vw]'></img>
        <h1 className='pt-[3vh] pb-[3vh] text-2xl text-center text-black font-bold hover:text-shadow-md'>Student Feedback Sentiment Analyzer</h1>
        <form onSubmit={handleSubmit} className='text-center ml-[1.25vw] mr-[1vw] h-full text-black'>
          <label className='text-black text-lg'>Enter your last score (out of 50)</label>
          <div className='ml-[1vw] mr-[1vw] *:border *:mt-5 *:mb-5 *:px-10 *:rounded-lg *:text-center *:focus:shadow-xl *:transition-all duration-200'>
          <input required value={score} onChange={(e) => setScore(e.target.value)} type="number" min="0" max="50" className=''></input>
          </div>
          <label className='text-black text-lg'>Enter your feedback about the course</label><br/>
          <textarea value={feedback} onChange={(e) => setFeedback(e.target.value)} className='w-75 text-center p-10 h-20 xl:w-100 xl:h-30 resize-none my-[1vh] border rounded-xl focus:shadow-xl focus-visible:none transition duration-200'></textarea><br/>
          <button type="submit" className='pl-5 pr-5 mb-5 rounded-xl text-xl font-["Consolas"] text-center hover:shadow-xl bg-blue-400 text-white active:scale-90 active:bg-gray-400 active:text-gray-500 active:shadow-none transition-all duration-50'>
          {loading ? "Analyzing..." : "Submit"}
          </button>
        </form>

        {result && (
          <div className='finalresult mt-6 p-4 bg-gray-600 rounded-xl'>
            <h2 className='text-lg font-semibold mb-2'>Result:</h2>
            <p><strong>Student Score:</strong> {score}/50</p>
            <p><strong>Sentiment of feedback:</strong> {result.sentiment}</p>
            <p><strong>Genuinity:</strong> {result.credibility_score*100}%</p>
            <p><strong>Interpretation:</strong> {result.flag}</p>
          </div>
        )}
      </div>
    </>
  );
}

export default App
