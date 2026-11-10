import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, CheckCircle2, Users, Bot, Sparkles, Brain, Send, AlertTriangle } from 'lucide-react';
import api from '../api/axios';

export default function TestResults() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [results, setResults] = useState(null);
  const [interventionDraft, setInterventionDraft] = useState(null);
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentApproving, setAgentApproving] = useState(false);
  const [agentMessage, setAgentMessage] = useState('');

  useEffect(() => {
    fetchResults();
  }, [id]);

  const fetchResults = async () => {
    try {
      const { data } = await api.get(`/teacher/tests/${id}/results`);
      setResults(data);
    } catch (e) {
      console.error(e);
    }
  };

  const runClassInterventionAgent = async () => {
    setAgentLoading(true);
    setAgentMessage('');
    try {
      const { data } = await api.post(`/agent/teacher/tests/${id}/intervention-draft`, {
        max_questions: 4
      });
      setInterventionDraft(data);
    } catch (e) {
      setAgentMessage(e.response?.data?.detail || 'Failed to run Class Intervention Agent.');
    } finally {
      setAgentLoading(false);
    }
  };

  const approveClassIntervention = async () => {
    if (!interventionDraft) return;
    const studentIds = (interventionDraft.struggling_students || []).map(s => s.student_id);
    if (studentIds.length === 0) {
      setAgentMessage('No struggling students found to assign this revision quiz.');
      return;
    }
    setAgentApproving(true);
    setAgentMessage('');
    try {
      const { data } = await api.post(`/agent/teacher/tests/${id}/intervention-approve`, {
        title: interventionDraft.draft_quiz_title,
        student_ids: studentIds,
        questions: interventionDraft.draft_questions
      });
      setAgentMessage(
        `Approved & assigned "${data.title}" (${data.question_count} questions) to ${data.assigned_student_ids.length} student(s).`
      );
    } catch (e) {
      setAgentMessage(e.response?.data?.detail || 'Failed to assign revision quiz.');
    } finally {
      setAgentApproving(false);
    }
  };

  if (!results) return <div className="text-center mt-20">Loading...</div>;

  const overallAccuracy = results.total_responses > 0 
    ? Math.round((results.total_correct / results.total_responses) * 100) 
    : 0;

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <button onClick={() => navigate('/teacher')} className="flex items-center gap-2 text-slate-500 hover:text-indigo-600 transition-colors">
        <ArrowLeft className="w-4 h-4" /> Back to Dashboard
      </button>

      <div>
        <h1 className="text-3xl font-bold text-slate-900">Test Results</h1>
        <p className="text-slate-500 mt-1">Overview of student performance</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-panel p-6 flex items-center gap-4">
          <div className="p-4 bg-indigo-50 text-indigo-600 rounded-2xl">
            <Users className="w-8 h-8" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-500 uppercase tracking-wider">Total Responses</p>
            <p className="text-3xl font-bold text-slate-900">{results.total_responses}</p>
          </div>
        </div>
        <div className="glass-panel p-6 flex items-center gap-4">
          <div className="p-4 bg-emerald-50 text-emerald-600 rounded-2xl">
            <CheckCircle2 className="w-8 h-8" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-500 uppercase tracking-wider">Overall Accuracy</p>
            <p className="text-3xl font-bold text-slate-900">{overallAccuracy}%</p>
          </div>
        </div>
      </div>

      <div className="glass-panel p-6">
        <h3 className="text-lg font-semibold text-slate-900 mb-6">Question Breakdown</h3>
        <div className="space-y-4">
          {Object.entries(results.question_stats).map(([qId, stats], idx) => (
            <div key={qId} className="flex items-center justify-between p-4 bg-slate-50 rounded-xl">
              <span className="font-medium text-slate-700">Question {idx + 1}</span>
              <div className="flex items-center gap-4">
                <span className="text-sm text-slate-500">{stats.correct} / {stats.total} correct</span>
                <div className="w-32 h-2.5 bg-slate-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-indigo-500" 
                    style={{ width: `${(stats.correct / stats.total) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
          {Object.keys(results.question_stats).length === 0 && (
            <div className="text-center text-slate-500 py-4">No responses yet.</div>
          )}
        </div>
      </div>

      {/* Class Intervention Agent Panel */}
      <div className="glass-panel p-6 border border-indigo-200 bg-gradient-to-br from-indigo-50/70 via-white to-slate-50 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="p-3 rounded-xl bg-indigo-600 text-white">
              <Bot className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-slate-900">Class Intervention AI Agent</h3>
              <p className="text-xs text-slate-600 mt-0.5">
                Diagnoses class-wide Q-matrix bottlenecks, rapid-guessing fatigue, and high-miss traps to draft a targeted revision quiz.
              </p>
            </div>
          </div>
          <button
            onClick={runClassInterventionAgent}
            disabled={agentLoading}
            className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-sm font-bold flex items-center gap-2 disabled:opacity-60"
          >
            <Sparkles className="w-4 h-4" />
            {agentLoading ? 'Analyzing Cohort...' : 'Analyze Class with AI Agent'}
          </button>
        </div>

        {agentMessage && (
          <div className="p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold">
            {agentMessage}
          </div>
        )}

        {interventionDraft && (
          <div className="space-y-4 pt-4 border-t border-indigo-100">
            <div className="p-4 rounded-xl bg-white border border-slate-200 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-black uppercase text-indigo-700">Teaching Briefing</span>
                <span className="font-mono text-slate-500">Engine: {interventionDraft.engine_mode}</span>
              </div>
              <p className="text-sm text-slate-800 font-medium">{interventionDraft.teaching_briefing}</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-white border border-slate-200 space-y-2">
                <p className="text-xs font-black uppercase text-rose-700 flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5" /> Concept Bottlenecks
                </p>
                {interventionDraft.bottleneck_concepts?.map((b, i) => (
                  <div key={i} className="flex justify-between text-xs py-1 border-b border-slate-100 last:border-0">
                    <span className="font-semibold text-slate-800">{b.concept_name}</span>
                    <span className="font-black text-rose-600">{b.error_rate_pct}% error</span>
                  </div>
                ))}
              </div>

              <div className="p-4 rounded-xl bg-slate-900 text-slate-100 space-y-1.5 font-mono text-[11px]">
                <p className="text-indigo-300 font-bold flex items-center gap-1.5">
                  <Brain className="w-3.5 h-3.5" /> Reasoning Trace
                </p>
                {interventionDraft.reasoning_trace?.map(st => (
                  <p key={st.step} className="text-slate-300">
                    [{st.step}] {st.tool_name}: {st.observation}
                  </p>
                ))}
              </div>
            </div>

            <div className="p-4 rounded-xl bg-white border border-indigo-200 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <span className="text-[10px] font-black uppercase text-indigo-600">Draft Revision Quiz Preview</span>
                  <h4 className="text-base font-bold text-slate-900">{interventionDraft.draft_quiz_title}</h4>
                  <p className="text-xs text-slate-500">
                    Target students ({interventionDraft.struggling_students?.length || 0}):{' '}
                    {interventionDraft.struggling_students?.map(s => `${s.name} (${s.accuracy_pct}%)`).join(', ')}
                  </p>
                </div>
                <button
                  onClick={approveClassIntervention}
                  disabled={agentApproving}
                  className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-black uppercase tracking-wider flex items-center gap-2 disabled:opacity-60"
                >
                  <Send className="w-3.5 h-3.5" />
                  {agentApproving ? 'Assigning...' : 'Approve & Assign Revision Quiz'}
                </button>
              </div>
              <div className="space-y-2">
                {interventionDraft.draft_questions?.map((dq, i) => (
                  <div key={i} className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs">
                    <p className="font-bold text-slate-800">Q{i + 1}. {dq.text}</p>
                    <p className="text-slate-500 mt-0.5">Skills: {dq.skill_names?.join(', ')}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
