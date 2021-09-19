public class Main 
{
	public static void main(String[] args)
	{
		try
		{			
			Terminal terminal1 = new Terminal("Broker");
			Terminal terminal2 = new Terminal("C&C");
			Terminal terminal3 = new Terminal("Worker");
			Terminal terminal4 = new Terminal("Worker2");
			Terminal terminal5 = new Terminal("C&C2");
			
			Broker broker = new Broker(terminal1, 50001);
			CommandAndControl CAndC = new CommandAndControl(terminal2, "localhost", 50001, 50000);
			Worker worker = new Worker(terminal3, "localhost", 50001, 50002);
			Worker worker2 = new Worker(terminal4, "localhost", 50001,50003);
			CommandAndControl CAndC2 = new CommandAndControl(terminal5, "localhost", 50001, 49999);
			
			Thread brokerThread = new Thread( broker );
			Thread CAndCThread = new Thread( CAndC );
			Thread workerThread = new Thread( worker );
			Thread workerThread2 = new Thread( worker2 );
			Thread CAndCThread2 = new Thread( CAndC2 );
			
			brokerThread.start();
			CAndCThread.start();
			workerThread.start();
			workerThread2.start();
			CAndCThread2.start();
		}
		catch(java.lang.Exception e) 
		{
			e.printStackTrace();
		}
	}
}
