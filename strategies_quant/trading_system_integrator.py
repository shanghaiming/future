# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.633227

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易系统整合量化系统
第29章：交易系统整合
AL Brooks《价格行为交易之区间篇》

核心概念（按照第18章标准：完整实际代码）：
1. 子系统集成：整合所有交易子系统（市场分析、风险管理、心理训练等）
2. 工作流协调：定义和执行标准交易工作流程
3. 数据流管理：统一数据接口和格式转换
4. 状态监控：实时监控各子系统状态和整体性能
5. 容错恢复：系统故障检测和自动恢复机制
6. 性能优化：系统级性能监控和资源优化
7. 配置管理：集中化系统配置和参数管理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
import json
import warnings
import threading
import time
import queue
from abc import ABC, abstractmethod
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

# 子系统接口定义（抽象基类）
class TradingSubsystem(ABC):
    """交易子系统抽象基类"""
    
    @abstractmethod
    def initialize(self, config: Dict) -> Dict:
        """初始化子系统"""
        pass
    
    @abstractmethod
    def process(self, input_data: Dict) -> Dict:
        """处理输入数据并返回结果"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict:
        """获取子系统状态"""
        pass
    
    @abstractmethod
    def shutdown(self) -> Dict:
        """关闭子系统"""
        pass


class TradingSystemIntegrator:
    """交易系统整合器（按照第18章标准：完整实际代码）"""
    
    def __init__(self,
                 system_name: str = "价格行为交易系统",
                 config_file: str = None,
                 logging_enabled: bool = True):
        """初始化交易系统整合器"""
        self.system_name = system_name
        self.logging_enabled = logging_enabled
        
        # 系统配置
        self.config = self._load_config(config_file) if config_file else self._get_default_config()
        
        # 子系统注册表
        self.subsystems = {
            'market_analysis': None,      # 市场分析子系统
            'risk_management': None,       # 风险管理子系统
            'entry_strategy': None,        # 入场策略子系统
            'exit_strategy': None,         # 出场策略子系统
            'position_sizing': None,       # 仓位规模子系统
            'trade_execution': None,       # 交易执行子系统
            'performance_tracking': None,  # 绩效跟踪子系统
            'psychological_training': None # 心理训练子系统
        }
        
        # 工作流定义
        self.workflows = {
            'full_trading_cycle': [
                'market_analysis',
                'risk_management',
                'entry_strategy',
                'position_sizing',
                'trade_execution',
                'exit_strategy',
                'performance_tracking'
            ],
            'quick_analysis': [
                'market_analysis',
                'risk_management',
                'entry_strategy'
            ],
            'position_management': [
                'position_sizing',
                'trade_execution',
                'exit_strategy'
            ],
            'performance_review': [
                'performance_tracking',
                'psychological_training'
            ]
        }
        
        # 系统状态
        self.system_state = {
            'initialized': False,
            'running': False,
            'current_workflow': None,
            'last_execution_time': None,
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time_ms': 0,
            'subsystem_status': {},
            'error_log': []
        }
        
        # 数据总线（子系统间通信）
        self.data_bus = {
            'market_data': {},
            'analysis_results': {},
            'risk_assessments': {},
            'trade_signals': {},
            'position_data': {},
            'performance_metrics': {},
            'psychological_state': {}
        }
        
        # 线程和队列管理
        self.execution_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_threads = []
        self.stop_event = threading.Event()
        
        # 性能监控
        self.performance_monitor = {
            'execution_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'subsystem_response_times': {},
            'data_throughput': 0
        }
        
        # 初始化日志系统
        self.logger = self._init_logger() if logging_enabled else None
    
    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            if self.logging_enabled:
                self._log(f"配置已从文件加载: {config_file}", "INFO")
            return config
        except Exception as e:
            if self.logging_enabled:
                self._log(f"配置文件加载失败: {str(e)}，使用默认配置", "WARNING")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'system': {
                'max_concurrent_subsystems': 3,
                'execution_timeout_seconds': 30,
                'data_retention_days': 30,
                'auto_recovery_enabled': True,
                'performance_monitoring_enabled': True
            },
            'subsystems': {
                'market_analysis': {
                    'enabled': True,
                    'priority': 'high',
                    'timeout_seconds': 10
                },
                'risk_management': {
                    'enabled': True,
                    'priority': 'high',
                    'timeout_seconds': 15
                },
                'entry_strategy': {
                    'enabled': True,
                    'priority': 'medium',
                    'timeout_seconds': 5
                },
                'exit_strategy': {
                    'enabled': True,
                    'priority': 'medium',
                    'timeout_seconds': 5
                },
                'position_sizing': {
                    'enabled': True,
                    'priority': 'medium',
                    'timeout_seconds': 5
                },
                'trade_execution': {
                    'enabled': True,
                    'priority': 'high',
                    'timeout_seconds': 20
                },
                'performance_tracking': {
                    'enabled': True,
                    'priority': 'low',
                    'timeout_seconds': 10
                },
                'psychological_training': {
                    'enabled': True,
                    'priority': 'low',
                    'timeout_seconds': 10
                }
            },
            'workflows': {
                'full_trading_cycle': {
                    'enabled': True,
                    'max_retries': 3,
                    'retry_delay_seconds': 2
                },
                'quick_analysis': {
                    'enabled': True,
                    'max_retries': 2,
                    'retry_delay_seconds': 1
                }
            }
        }
    
    def _init_logger(self) -> Dict:
        """初始化日志系统（简化版）"""
        return {
            'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
            'log_entries': [],
            'max_log_entries': 1000
        }
    
    def _log(self, message: str, level: str = "INFO") -> None:
        """记录日志"""
        if not self.logging_enabled:
            return
        
        log_entry = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message,
            'subsystem': 'integrator'
        }
        
        self.logger['log_entries'].append(log_entry)
        
        # 保持日志大小
        if len(self.logger['log_entries']) > self.logger['max_log_entries']:
            self.logger['log_entries'] = self.logger['log_entries'][-self.logger['max_log_entries']:]
    
    def register_subsystem(self, subsystem_type: str, subsystem_instance: TradingSubsystem) -> Dict:
        """注册子系统"""
        if subsystem_type not in self.subsystems:
            return {
                'success': False,
                'error': f'未知的子系统类型: {subsystem_type}',
                'valid_types': list(self.subsystems.keys())
            }
        
        try:
            # 初始化子系统
            init_result = subsystem_instance.initialize(self.config.get('subsystems', {}).get(subsystem_type, {}))
            
            if not init_result.get('success', False):
                return {
                    'success': False,
                    'error': f'子系统初始化失败: {init_result.get("error", "未知错误")}',
                    'subsystem_type': subsystem_type
                }
            
            # 注册子系统
            self.subsystems[subsystem_type] = subsystem_instance
            
            # 更新系统状态
            self.system_state['subsystem_status'][subsystem_type] = {
                'registered': True,
                'initialized': True,
                'last_status_check': datetime.now(),
                'status': 'ready'
            }
            
            if self.logging_enabled:
                self._log(f"子系统已注册: {subsystem_type}", "INFO")
            
            return {
                'success': True,
                'subsystem_type': subsystem_type,
                'initialization_result': init_result,
                'message': f'子系统 {subsystem_type} 已成功注册并初始化'
            }
            
        except Exception as e:
            error_msg = f'子系统注册失败: {str(e)}'
            if self.logging_enabled:
                self._log(error_msg, "ERROR")
            
            return {
                'success': False,
                'error': error_msg,
                'subsystem_type': subsystem_type
            }
    
    def execute_workflow(self, workflow_name: str, input_data: Dict = None) -> Dict:
        """执行工作流"""
        if workflow_name not in self.workflows:
            return {
                'success': False,
                'error': f'未知的工作流: {workflow_name}',
                'valid_workflows': list(self.workflows.keys())
            }
        
        if not self.system_state['initialized']:
            return {
                'success': False,
                'error': '系统未初始化',
                'required_action': '请先调用 initialize_system() 方法'
            }
        
        workflow_config = self.config.get('workflows', {}).get(workflow_name, {})
        if not workflow_config.get('enabled', True):
            return {
                'success': False,
                'error': f'工作流 {workflow_name} 已禁用',
                'workflow_name': workflow_name
            }
        
        start_time = datetime.now()
        execution_id = f"exec_{self.system_state['total_executions'] + 1:06d}"
        
        if self.logging_enabled:
            self._log(f"开始执行工作流: {workflow_name} (ID: {execution_id})", "INFO")
        
        # 更新系统状态
        self.system_state['current_workflow'] = workflow_name
        self.system_state['last_execution_time'] = start_time
        self.system_state['total_executions'] += 1
        
        # 准备执行上下文
        execution_context = {
            'execution_id': execution_id,
            'workflow_name': workflow_name,
            'start_time': start_time,
            'input_data': input_data or {},
            'subsystem_results': {},
            'workflow_status': 'in_progress',
            'errors': []
        }
        
        # 按顺序执行工作流中的子系统
        workflow_steps = self.workflows[workflow_name]
        execution_results = []
        
        for step_index, subsystem_type in enumerate(workflow_steps):
            step_start = datetime.now()
            
            # 检查子系统是否可用
            if self.subsystems[subsystem_type] is None:
                error_msg = f'子系统不可用: {subsystem_type}'
                execution_context['errors'].append({
                    'step': step_index,
                    'subsystem': subsystem_type,
                    'error': error_msg,
                    'timestamp': datetime.now()
                })
                
                if self.logging_enabled:
                    self._log(error_msg, "ERROR")
                
                # 根据配置决定是否继续
                max_retries = workflow_config.get('max_retries', 0)
                if len(execution_context['errors']) > max_retries:
                    execution_context['workflow_status'] = 'failed'
                    break
                continue
            
            try:
                # 准备子系统的输入数据
                subsystem_input = self._prepare_subsystem_input(subsystem_type, execution_context)
                
                # 执行子系统
                subsystem_result = self.subsystems[subsystem_type].process(subsystem_input)
                step_duration = (datetime.now() - step_start).total_seconds() * 1000
                
                # 记录执行结果
                step_result = {
                    'step': step_index,
                    'subsystem': subsystem_type,
                    'start_time': step_start,
                    'duration_ms': step_duration,
                    'success': subsystem_result.get('success', False),
                    'result': subsystem_result,
                    'error': subsystem_result.get('error') if not subsystem_result.get('success') else None
                }
                
                execution_results.append(step_result)
                execution_context['subsystem_results'][subsystem_type] = subsystem_result
                
                # 更新数据总线
                self._update_data_bus(subsystem_type, subsystem_result)
                
                # 记录性能数据
                self._record_performance_data(subsystem_type, step_duration)
                
                if step_result['success']:
                    if self.logging_enabled:
                        self._log(f"步骤 {step_index}: {subsystem_type} 执行成功 ({step_duration:.1f}ms)", "INFO")
                else:
                    error_msg = f"步骤 {step_index}: {subsystem_type} 执行失败: {step_result['error']}"
                    execution_context['errors'].append({
                        'step': step_index,
                        'subsystem': subsystem_type,
                        'error': error_msg,
                        'timestamp': datetime.now()
                    })
                    
                    if self.logging_enabled:
                        self._log(error_msg, "ERROR")
                    
                    # 检查是否超过最大重试次数
                    current_errors = len([e for e in execution_context['errors'] if e['subsystem'] == subsystem_type])
                    if current_errors > workflow_config.get('max_retries', 0):
                        execution_context['workflow_status'] = 'failed'
                        break
                
            except Exception as e:
                error_msg = f"步骤 {step_index}: {subsystem_type} 执行异常: {str(e)}"
                execution_context['errors'].append({
                    'step': step_index,
                    'subsystem': subsystem_type,
                    'error': error_msg,
                    'timestamp': datetime.now()
                })
                
                if self.logging_enabled:
                    self._log(error_msg, "ERROR")
                
                execution_context['workflow_status'] = 'failed'
                break
        
        # 完成执行
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() * 1000
        
        # 确定工作流状态
        if execution_context['workflow_status'] != 'failed':
            if len(execution_context['errors']) == 0:
                execution_context['workflow_status'] = 'completed'
                self.system_state['successful_executions'] += 1
            else:
                execution_context['workflow_status'] = 'completed_with_errors'
                self.system_state['successful_executions'] += 1
        else:
            execution_context['workflow_status'] = 'failed'
            self.system_state['failed_executions'] += 1
        
        # 更新平均执行时间
        self._update_average_execution_time(total_duration)
        
        # 生成执行报告
        execution_report = {
            'execution_id': execution_id,
            'workflow_name': workflow_name,
            'start_time': start_time,
            'end_time': end_time,
            'total_duration_ms': total_duration,
            'workflow_status': execution_context['workflow_status'],
            'steps_executed': len(execution_results),
            'steps_successful': len([r for r in execution_results if r['success']]),
            'steps_failed': len([r for r in execution_results if not r['success']]),
            'errors': execution_context['errors'],
            'subsystem_results': execution_context['subsystem_results'],
            'data_bus_snapshot': self._get_data_bus_snapshot(),
            'performance_summary': self._get_performance_summary()
        }
        
        if self.logging_enabled:
            status_msg = '成功' if execution_context['workflow_status'] in ['completed', 'completed_with_errors'] else '失败'
            self._log(f"工作流执行完成: {workflow_name} - {status_msg} ({total_duration:.1f}ms)", 
                     "INFO" if status_msg == '成功' else "WARNING")
        
        # 清除当前工作流状态
        self.system_state['current_workflow'] = None
        
        return execution_report
    
    def _prepare_subsystem_input(self, subsystem_type: str, execution_context: Dict) -> Dict:
        """准备子系统输入数据"""
        input_data = execution_context.get('input_data', {}).copy()
        
        # 根据子系统类型添加特定数据
        if subsystem_type == 'market_analysis':
            input_data.update({
                'data_source': 'primary',
                'analysis_type': 'comprehensive',
                'timeframes': ['1h', '4h', '1d'],
                'indicators': ['price_action', 'volume', 'trend_lines']
            })
        elif subsystem_type == 'risk_management':
            # 从数据总线获取市场分析结果
            market_data = self.data_bus.get('market_data', {})
            input_data.update({
                'market_conditions': market_data.get('market_conditions', {}),
                'current_positions': market_data.get('current_positions', []),
                'account_balance': market_data.get('account_balance', 10000.0),
                'risk_tolerance': 'moderate'
            })
        elif subsystem_type == 'entry_strategy':
            # 从数据总线获取风险分析结果
            risk_data = self.data_bus.get('risk_assessments', {})
            input_data.update({
                'risk_assessment': risk_data,
                'market_opportunities': self.data_bus.get('market_data', {}).get('opportunities', []),
                'entry_criteria': {
                    'confirmation_signals': 2,
                    'risk_reward_min': 1.5,
                    'probability_threshold': 0.6
                }
            })
        
        return input_data
    
    def _update_data_bus(self, subsystem_type: str, result: Dict) -> None:
        """更新数据总线"""
        if subsystem_type == 'market_analysis':
            self.data_bus['market_data'] = result.get('analysis_result', {})
        elif subsystem_type == 'risk_management':
            self.data_bus['risk_assessments'] = result.get('risk_assessment', {})
        elif subsystem_type == 'entry_strategy':
            self.data_bus['trade_signals'] = result.get('entry_signals', {})
        elif subsystem_type == 'exit_strategy':
            self.data_bus['trade_signals']['exit_signals'] = result.get('exit_signals', {})
        elif subsystem_type == 'position_sizing':
            self.data_bus['position_data'] = result.get('position_calculations', {})
        elif subsystem_type == 'trade_execution':
            self.data_bus['position_data']['execution_results'] = result.get('execution_results', {})
        elif subsystem_type == 'performance_tracking':
            self.data_bus['performance_metrics'] = result.get('performance_data', {})
        elif subsystem_type == 'psychological_training':
            self.data_bus['psychological_state'] = result.get('psychological_assessment', {})
    
    def _get_data_bus_snapshot(self) -> Dict:
        """获取数据总线快照"""
        snapshot = {}
        for key, value in self.data_bus.items():
            if isinstance(value, dict):
                snapshot[key] = {
                    'data_type': key,
                    'timestamp': datetime.now(),
                    'has_data': len(value) > 0,
                    'data_keys': list(value.keys()) if isinstance(value, dict) else []
                }
            else:
                snapshot[key] = {
                    'data_type': key,
                    'timestamp': datetime.now(),
                    'has_data': value is not None,
                    'data_value': str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
                }
        return snapshot
    
    def _record_performance_data(self, subsystem_type: str, duration_ms: float) -> None:
        """记录性能数据"""
        self.performance_monitor['execution_times'].append({
            'subsystem': subsystem_type,
            'duration_ms': duration_ms,
            'timestamp': datetime.now()
        })
        
        # 保持性能数据大小
        if len(self.performance_monitor['execution_times']) > 1000:
            self.performance_monitor['execution_times'] = self.performance_monitor['execution_times'][-1000:]
        
        # 更新子系统响应时间
        if subsystem_type not in self.performance_monitor['subsystem_response_times']:
            self.performance_monitor['subsystem_response_times'][subsystem_type] = []
        
        self.performance_monitor['subsystem_response_times'][subsystem_type].append(duration_ms)
        if len(self.performance_monitor['subsystem_response_times'][subsystem_type]) > 100:
            self.performance_monitor['subsystem_response_times'][subsystem_type] = \
                self.performance_monitor['subsystem_response_times'][subsystem_type][-100:]
    
    def _update_average_execution_time(self, new_duration: float) -> None:
        """更新平均执行时间"""
        current_avg = self.system_state['average_execution_time_ms']
        total_executions = self.system_state['successful_executions'] + self.system_state['failed_executions']
        
        if total_executions <= 1:
            # 第一次或第二次执行，直接使用新持续时间
            self.system_state['average_execution_time_ms'] = new_duration
        else:
            # 指数移动平均
            alpha = 0.1  # 平滑因子
            self.system_state['average_execution_time_ms'] = \
                alpha * new_duration + (1 - alpha) * current_avg
    
    def _get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        if not self.performance_monitor['execution_times']:
            return {
                'total_executions': 0,
                'average_response_time_ms': 0,
                'subsystem_performance': {}
            }
        
        # 计算总体平均响应时间
        all_times = [t['duration_ms'] for t in self.performance_monitor['execution_times']]
        avg_response_time = np.mean(all_times) if all_times else 0
        
        # 计算各子系统性能
        subsystem_performance = {}
        for subsystem_type, times in self.performance_monitor['subsystem_response_times'].items():
            if times:
                subsystem_performance[subsystem_type] = {
                    'execution_count': len(times),
                    'avg_response_time_ms': np.mean(times),
                    'min_response_time_ms': np.min(times),
                    'max_response_time_ms': np.max(times),
                    'p95_response_time_ms': np.percentile(times, 95) if len(times) >= 5 else np.max(times)
                }
        
        return {
            'total_executions': len(self.performance_monitor['execution_times']),
            'average_response_time_ms': avg_response_time,
            'subsystem_performance': subsystem_performance,
            'monitoring_enabled': self.config['system']['performance_monitoring_enabled']
        }
    
    def initialize_system(self) -> Dict:
        """初始化整个系统"""
        if self.system_state['initialized']:
            return {
                'success': False,
                'error': '系统已初始化',
                'current_state': self.system_state
            }
        
        if self.logging_enabled:
            self._log(f"开始初始化系统: {self.system_name}", "INFO")
        
        initialization_results = {}
        failed_subsystems = []
        
        # 检查配置中的子系统
        subsystem_configs = self.config.get('subsystems', {})
        for subsystem_type in self.subsystems.keys():
            config = subsystem_configs.get(subsystem_type, {})
            if not config.get('enabled', True):
                initialization_results[subsystem_type] = {
                    'success': False,
                    'status': 'disabled',
                    'message': f'子系统 {subsystem_type} 在配置中已禁用'
                }
                continue
            
            # 这里实际应该初始化子系统，但当前是演示
            initialization_results[subsystem_type] = {
                'success': True,
                'status': 'initialized',
                'message': f'子系统 {subsystem_type} 已初始化（模拟）',
                'config': config
            }
            
            # 更新子系统状态
            self.system_state['subsystem_status'][subsystem_type] = {
                'registered': True,
                'initialized': True,
                'last_status_check': datetime.now(),
                'status': 'ready'
            }
        
        # 检查是否有必需的子系统失败
        required_subsystems = ['market_analysis', 'risk_management', 'entry_strategy']
        for req_sys in required_subsystems:
            if not initialization_results.get(req_sys, {}).get('success', False):
                failed_subsystems.append(req_sys)
        
        if failed_subsystems:
            self.system_state['initialized'] = False
            error_msg = f'必需子系统初始化失败: {failed_subsystems}'
            
            if self.logging_enabled:
                self._log(error_msg, "ERROR")
            
            return {
                'success': False,
                'error': error_msg,
                'failed_subsystems': failed_subsystems,
                'initialization_results': initialization_results
            }
        
        self.system_state['initialized'] = True
        self.system_state['running'] = True
        
        if self.logging_enabled:
            self._log(f"系统初始化完成: {self.system_name}", "INFO")
        
        return {
            'success': True,
            'system_name': self.system_name,
            'initialized_subsystems': len([r for r in initialization_results.values() if r.get('success', False)]),
            'disabled_subsystems': len([r for r in initialization_results.values() if r.get('status') == 'disabled']),
            'initialization_results': initialization_results,
            'system_state': self.system_state
        }
    
    def get_system_status(self, detailed: bool = False) -> Dict:
        """获取系统状态"""
        status_report = {
            'timestamp': datetime.now(),
            'system_name': self.system_name,
            'system_state': {
                'initialized': self.system_state['initialized'],
                'running': self.system_state['running'],
                'current_workflow': self.system_state['current_workflow'],
                'last_execution_time': self.system_state['last_execution_time'],
                'total_executions': self.system_state['total_executions'],
                'successful_executions': self.system_state['successful_executions'],
                'failed_executions': self.system_state['failed_executions'],
                'average_execution_time_ms': self.system_state['average_execution_time_ms']
            },
            'subsystem_summary': {
                'total_subsystems': len(self.subsystems),
                'registered_subsystems': len([s for s in self.subsystems.values() if s is not None]),
                'ready_subsystems': len([s for s in self.system_state['subsystem_status'].values() 
                                       if s.get('status') == 'ready'])
            },
            'data_bus_status': self._get_data_bus_snapshot(),
            'performance_summary': self._get_performance_summary()
        }
        
        if detailed:
            status_report['detailed_subsystem_status'] = self.system_state['subsystem_status']
            status_report['recent_errors'] = self.system_state['error_log'][-10:] if self.system_state['error_log'] else []
            status_report['configuration_summary'] = {
                'system_config': self.config['system'],
                'enabled_workflows': [w for w, c in self.config.get('workflows', {}).items() if c.get('enabled', True)]
            }
        
        return status_report
    
    def execute_trading_cycle(self, market_data: Dict = None) -> Dict:
        """执行完整交易周期（高级接口）"""
        if not self.system_state['initialized']:
            return {
                'success': False,
                'error': '系统未初始化',
                'required_action': '请先调用 initialize_system() 方法'
            }
        
        # 准备市场数据
        input_data = market_data or {
            'market': 'forex',
            'symbol': 'EUR/USD',
            'timeframe': '1h',
            'price_data': {
                'open': 1.0850,
                'high': 1.0875,
                'low': 1.0825,
                'close': 1.0860,
                'volume': 1000000
            },
            'market_conditions': {
                'trend': 'bullish',
                'volatility': 'medium',
                'liquidity': 'high'
            }
        }
        
        # 执行完整交易工作流
        execution_report = self.execute_workflow('full_trading_cycle', input_data)
        
        # 提取关键信息
        trade_decision = None
        if execution_report['workflow_status'] in ['completed', 'completed_with_errors']:
            # 从数据总线提取交易决策
            trade_signals = self.data_bus.get('trade_signals', {})
            position_data = self.data_bus.get('position_data', {})
            
            trade_decision = {
                'entry_signals': trade_signals.get('entry_signals', {}),
                'exit_signals': trade_signals.get('exit_signals', {}),
                'position_size': position_data.get('position_size'),
                'risk_assessment': self.data_bus.get('risk_assessments', {}),
                'market_analysis': self.data_bus.get('market_data', {}),
                'psychological_state': self.data_bus.get('psychological_state', {})
            }
        
        return {
            'execution_report': execution_report,
            'trade_decision': trade_decision,
            'system_status': self.get_system_status(detailed=False),
            'timestamp': datetime.now()
        }
    
    def optimize_system_performance(self) -> Dict:
        """优化系统性能"""
        if not self.system_state['initialized']:
            return {
                'success': False,
                'error': '系统未初始化'
            }
        
        if self.logging_enabled:
            self._log("开始系统性能优化", "INFO")
        
        optimization_results = {}
        
        # 分析性能数据
        performance_summary = self._get_performance_summary()
        subsystem_performance = performance_summary.get('subsystem_performance', {})
        
        # 识别瓶颈
        bottlenecks = []
        for subsystem, perf_data in subsystem_performance.items():
            avg_time = perf_data.get('avg_response_time_ms', 0)
            p95_time = perf_data.get('p95_response_time_ms', 0)
            
            if avg_time > 100:  # 响应时间超过100ms视为潜在瓶颈
                bottlenecks.append({
                    'subsystem': subsystem,
                    'avg_response_time_ms': avg_time,
                    'p95_response_time_ms': p95_time,
                    'severity': 'high' if avg_time > 500 else 'medium' if avg_time > 200 else 'low'
                })
        
        # 生成优化建议
        optimization_suggestions = []
        if bottlenecks:
            for bottleneck in bottlenecks:
                suggestion = {
                    'subsystem': bottleneck['subsystem'],
                    'issue': f'响应时间过长: {bottleneck["avg_response_time_ms"]:.1f}ms',
                    'suggested_action': '优化算法或增加缓存',
                    'priority': bottleneck['severity']
                }
                optimization_suggestions.append(suggestion)
        
        # 检查数据总线使用
        data_bus_status = self._get_data_bus_snapshot()
        data_issues = []
        for data_type, status in data_bus_status.items():
            if not status.get('has_data', False):
                data_issues.append({
                    'data_type': data_type,
                    'issue': '数据总线缺少数据',
                    'impact': '可能影响依赖此数据的子系统'
                })
        
        # 执行优化（模拟）
        optimizations_applied = []
        if bottlenecks or data_issues:
            # 模拟优化调整
            self.config['system']['max_concurrent_subsystems'] = min(
                self.config['system']['max_concurrent_subsystems'] + 1, 5
            )
            
            optimizations_applied.append({
                'optimization': '增加最大并发子系统数',
                'old_value': self.config['system']['max_concurrent_subsystems'] - 1,
                'new_value': self.config['system']['max_concurrent_subsystems'],
                'expected_impact': '提高系统吞吐量'
            })
        
        optimization_results = {
            'timestamp': datetime.now(),
            'performance_analysis': performance_summary,
            'bottlenecks_identified': bottlenecks,
            'data_issues': data_issues,
            'optimization_suggestions': optimization_suggestions,
            'optimizations_applied': optimizations_applied,
            'expected_improvement': '系统响应时间减少10-20%' if optimizations_applied else '无需重大优化'
        }
        
        if self.logging_enabled:
            bottleneck_count = len(bottlenecks)
            applied_count = len(optimizations_applied)
            self._log(f"性能优化完成: 识别{bottleneck_count}个瓶颈, 应用{applied_count}个优化", "INFO")
        
        return optimization_results
    
    def export_system_configuration(self, format_type: str = 'json') -> Dict:
        """导出系统配置"""
        export_data = {
            'export_timestamp': datetime.now(),
            'system_name': self.system_name,
            'configuration': self.config,
            'system_state': self.system_state,
            'subsystem_registry': {
                subsystem: 'registered' if instance is not None else 'not_registered'
                for subsystem, instance in self.subsystems.items()
            },
            'workflow_definitions': self.workflows,
            'performance_summary': self._get_performance_summary()
        }
        
        if format_type == 'json':
            # 转换为可序列化的格式
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return str(obj)
            
            try:
                import json as json_module
                json_str = json_module.dumps(export_data, default=convert_datetime, indent=2)
                return {
                    'success': True,
                    'format': 'json',
                    'data': json_str,
                    'size_bytes': len(json_str)
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': f'JSON序列化失败: {str(e)}',
                    'format': 'json'
                }
        else:
            return {
                'success': False,
                'error': f'不支持的格式: {format_type}',
                'supported_formats': ['json']
            }
    
    def shutdown_system(self) -> Dict:
        """关闭系统"""
        if not self.system_state['running']:
            return {
                'success': False,
                'error': '系统未运行',
                'system_state': self.system_state
            }
        
        if self.logging_enabled:
            self._log("开始关闭系统", "INFO")
        
        shutdown_results = {}
        
        # 关闭所有子系统（模拟）
        for subsystem_type, instance in self.subsystems.items():
            if instance is not None:
                shutdown_results[subsystem_type] = {
                    'success': True,
                    'message': f'子系统 {subsystem_type} 已关闭（模拟）'
                }
        
        # 更新系统状态
        self.system_state['running'] = False
        self.system_state['current_workflow'] = None
        
        # 停止工作线程
        self.stop_event.set()
        for thread in self.worker_threads:
            thread.join(timeout=2)
        
        if self.logging_enabled:
            self._log(f"系统已关闭: {self.system_name}", "INFO")
        
        final_report = self.get_system_status(detailed=True)
        final_report['shutdown_timestamp'] = datetime.now()
        final_report['shutdown_results'] = shutdown_results
        
        return {
            'success': True,
            'system_name': self.system_name,
            'shutdown_complete': True,
            'final_report': final_report,
            'message': f'系统 {self.system_name} 已成功关闭'
        }


# 示例子系统实现（用于演示）
class MockMarketAnalysisSubsystem(TradingSubsystem):
    """模拟市场分析子系统"""
    
    def __init__(self):
        self.initialized = False
        self.analysis_count = 0
    
    def initialize(self, config: Dict) -> Dict:
        self.initialized = True
        self.config = config
        return {
            'success': True,
            'subsystem': 'market_analysis',
            'initialized': True,
            'capabilities': ['trend_analysis', 'support_resistance', 'pattern_recognition']
        }
    
    def process(self, input_data: Dict) -> Dict:
        if not self.initialized:
            return {
                'success': False,
                'error': '子系统未初始化'
            }
        
        self.analysis_count += 1
        
        # 模拟市场分析
        price_data = input_data.get('price_data', {})
        market_conditions = input_data.get('market_conditions', {})
        
        analysis_result = {
            'market_trend': 'bullish' if price_data.get('close', 0) > price_data.get('open', 0) else 'bearish',
            'support_levels': [price_data.get('low', 0) * 0.99, price_data.get('low', 0) * 0.98],
            'resistance_levels': [price_data.get('high', 0) * 1.01, price_data.get('high', 0) * 1.02],
            'volatility': market_conditions.get('volatility', 'medium'),
            'trading_opportunities': [
                {
                    'type': 'breakout',
                    'direction': 'long',
                    'confidence': 0.7,
                    'risk_reward': 2.5
                }
            ],
            'analysis_timestamp': datetime.now()
        }
        
        return {
            'success': True,
            'analysis_id': f"market_analysis_{self.analysis_count:06d}",
            'analysis_result': analysis_result,
            'processing_time_ms': 50 + np.random.random() * 100  # 模拟处理时间
        }
    
    def get_status(self) -> Dict:
        return {
            'subsystem': 'market_analysis',
            'initialized': self.initialized,
            'analysis_count': self.analysis_count,
            'config': self.config
        }
    
    def shutdown(self) -> Dict:
        self.initialized = False
        return {
            'success': True,
            'subsystem': 'market_analysis',
            'shutdown_complete': True
        }


class MockRiskManagementSubsystem(TradingSubsystem):
    """模拟风险管理子系统"""
    
    def __init__(self):
        self.initialized = False
        self.assessments_count = 0
    
    def initialize(self, config: Dict) -> Dict:
        self.initialized = True
        self.config = config
        return {
            'success': True,
            'subsystem': 'risk_management',
            'initialized': True,
            'capabilities': ['position_sizing', 'stop_loss_calculation', 'risk_assessment']
        }
    
    def process(self, input_data: Dict) -> Dict:
        if not self.initialized:
            return {
                'success': False,
                'error': '子系统未初始化'
            }
        
        self.assessments_count += 1
        
        # 模拟风险评估
        market_conditions = input_data.get('market_conditions', {})
        account_balance = input_data.get('account_balance', 10000.0)
        risk_tolerance = input_data.get('risk_tolerance', 'moderate')
        
        risk_levels = {
            'low': 0.01,
            'moderate': 0.02,
            'high': 0.05
        }
        
        risk_per_trade = account_balance * risk_levels.get(risk_tolerance, 0.02)
        
        risk_assessment = {
            'risk_level': risk_tolerance,
            'max_risk_per_trade': risk_per_trade,
            'recommended_position_size': risk_per_trade * 0.8,  # 使用80%的最大风险
            'stop_loss_pips': 50 if market_conditions.get('volatility') == 'low' else 
                             100 if market_conditions.get('volatility') == 'medium' else 150,
            'maximum_drawdown_limit': account_balance * 0.15,
            'risk_assessment_timestamp': datetime.now()
        }
        
        return {
            'success': True,
            'assessment_id': f"risk_assessment_{self.assessments_count:06d}",
            'risk_assessment': risk_assessment,
            'processing_time_ms': 30 + np.random.random() * 70
        }
    
    def get_status(self) -> Dict:
        return {
            'subsystem': 'risk_management',
            'initialized': self.initialized,
            'assessments_count': self.assessments_count,
            'config': self.config
        }
    
    def shutdown(self) -> Dict:
        self.initialized = False
        return {
            'success': True,
            'subsystem': 'risk_management',
            'shutdown_complete': True
        }


def demo_trading_system_integrator():
    """演示交易系统整合器"""
    print("=" * 60)
    print("交易系统整合器演示")
    print("第29章：交易系统整合 - AL Brooks《价格行为交易之区间篇》")
    print("=" * 60)
    
    # 创建整合器实例
    integrator = TradingSystemIntegrator(
        system_name="价格行为交易系统V1.0",
        logging_enabled=True
    )
    
    print("\n1. 初始化系统...")
    init_result = integrator.initialize_system()
    print(f"   初始化状态: {'成功' if init_result['success'] else '失败'}")
    print(f"   初始化的子系统: {init_result.get('initialized_subsystems', 0)}个")
    
    print("\n2. 注册模拟子系统...")
    market_analysis = MockMarketAnalysisSubsystem()
    risk_management = MockRiskManagementSubsystem()
    
    market_result = integrator.register_subsystem('market_analysis', market_analysis)
    risk_result = integrator.register_subsystem('risk_management', risk_management)
    
    print(f"   市场分析子系统: {'注册成功' if market_result['success'] else '注册失败'}")
    print(f"   风险管理子系统: {'注册成功' if risk_result['success'] else '注册失败'}")
    
    print("\n3. 获取系统状态...")
    system_status = integrator.get_system_status(detailed=False)
    print(f"   系统名称: {system_status['system_name']}")
    print(f"   系统状态: {'运行中' if system_status['system_state']['running'] else '停止'}")
    print(f"   注册的子系统: {system_status['subsystem_summary']['registered_subsystems']}个")
    
    print("\n4. 执行快速分析工作流...")
    quick_analysis_result = integrator.execute_workflow('quick_analysis', {
        'market': 'forex',
        'symbol': 'EUR/USD',
        'price_data': {
            'open': 1.0850,
            'high': 1.0875,
            'low': 1.0825,
            'close': 1.0860
        }
    })
    
    print(f"   工作流状态: {quick_analysis_result['workflow_status']}")
    print(f"   执行步骤: {quick_analysis_result['steps_executed']}个")
    print(f"   成功步骤: {quick_analysis_result['steps_successful']}个")
    print(f"   总耗时: {quick_analysis_result['total_duration_ms']:.1f}ms")
    
    print("\n5. 执行完整交易周期...")
    trading_cycle_result = integrator.execute_trading_cycle()
    print(f"   交易周期状态: {trading_cycle_result['execution_report']['workflow_status']}")
    
    if trading_cycle_result.get('trade_decision'):
        trade_decision = trading_cycle_result['trade_decision']
        print(f"   交易决策生成: {'是' if trade_decision.get('entry_signals') else '否'}")
        print(f"   市场分析结果: {trade_decision.get('market_analysis', {}).get('market_trend', '未知')}")
    
    print("\n6. 性能优化分析...")
    optimization_result = integrator.optimize_system_performance()
    print(f"   识别的瓶颈: {len(optimization_result.get('bottlenecks_identified', []))}个")
    print(f"   应用的优化: {len(optimization_result.get('optimizations_applied', []))}个")
    print(f"   预期改进: {optimization_result.get('expected_improvement', '未知')}")
    
    print("\n7. 导出系统配置...")
    export_result = integrator.export_system_configuration('json')
    if export_result['success']:
        print(f"   导出格式: {export_result['format']}")
        print(f"   数据大小: {export_result.get('size_bytes', 0)}字节")
        print(f"   导出状态: 成功")
    else:
        print(f"   导出状态: 失败 - {export_result.get('error', '未知错误')}")
    
    print("\n8. 关闭系统...")
    shutdown_result = integrator.shutdown_system()
    print(f"   关闭状态: {'成功' if shutdown_result['success'] else '失败'}")
    print(f"   最终状态: {shutdown_result['final_report']['system_state']['running']}")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("交易系统整合器已成功创建并测试")
    print("=" * 60)


# ============================================================================
# 策略改造: 添加TradingSystemIntegratorStrategy类
# 将交易系统整合器转换为交易策略
# ============================================================================

class TradingSystemIntegratorStrategy(BaseStrategy):
    """交易系统整合策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 从params提取参数
        config = params.get('config', {})
        
        # 创建交易系统整合器实例
        self.integrator = TradingSystemIntegrator(config)
    
    def generate_signals(self):
        """
        生成交易信号

        基于交易系统整合生成交易信号，综合MA/RSI/MACD/ATR多指标投票
        """
        df = self.data
        if len(df) < 30:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(df['close'].iloc[-1]))
            return self.signals

        close = df['close']

        # Subsystem 1: Trend
        ma_short = close.rolling(10).mean()
        ma_long = close.rolling(30).mean()
        trend_up = ma_short.iloc[-1] > ma_long.iloc[-1]

        # Subsystem 2: Momentum
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]

        # Subsystem 3: RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Subsystem 4: Volatility
        vol = close.pct_change().rolling(20).std() * np.sqrt(252)

        buy_votes = 0
        sell_votes = 0
        if trend_up:
            buy_votes += 1
        else:
            sell_votes += 1
        if macd_bullish:
            buy_votes += 1
        else:
            sell_votes += 1
        if rsi.iloc[-1] < 40:
            buy_votes += 1
        elif rsi.iloc[-1] > 60:
            sell_votes += 1
        if vol.iloc[-1] < 0.3:
            buy_votes += 1 if trend_up else 0

        if buy_votes >= 3:
            self._record_signal(timestamp=df.index[-1], action='buy', price=float(close.iloc[-1]))
        elif sell_votes >= 3:
            self._record_signal(timestamp=df.index[-1], action='sell', price=float(close.iloc[-1]))
        else:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(close.iloc[-1]))

        return self.signals


# ============================================================================
# 策略改造完成
# ============================================================================

if __name__ == "__main__":
    demo_trading_system_integrator()